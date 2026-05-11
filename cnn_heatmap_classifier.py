#!/usr/bin/env python3
"""
V9: CNN热力图情绪分类器
========================
思路: 将468个3D landmarks → 热力图像 → 轻量CNN → 7类情绪
优势: 
  - 保留完整的空间结构信息(14维特征丢失了大量空间关系)
  - CNN自动学习局部模式(如眉眼组合、嘴部形状)
  - 数据增强简单(旋转/翻转热力图)
  
作者: AI | 日期: 2026-05-08
"""
import os, sys, time, json, glob, pickle, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================== 配置 ====================
BASE = 'RAF-DB/train'   # 方案B: PC端使用RAF-DB全量数据
CATEGORIES = {
    '0惊讶':'惊讶','1恐惧':'恐惧','2厌恶':'厌恶',
    '3快乐':'开心','4悲伤':'悲伤','5愤怒':'愤怒','6中性':'中性'
}
LABEL_ORDER = ['惊讶','恐惧','厌恶','开心','悲伤','愤怒','中性']
LABEL2IDX = {e:i for i,e in enumerate(LABEL_ORDER)}

HEATMAP_SIZE = 128       # 热力图尺寸 (方案B: PC端128x128高分辨率)
GAUSS_SIGMA = 1.5        # 高斯核标准差(相对于图大小)
BATCH_SIZE = 64
EPOCHS = 200              # V11: 给足够时间+OneCycleLR自动调度
LR = 1e-3                 # V11: 回归保守初始LR
WEIGHT_DECAY = 1e-3        # V11: 更强L2正则化

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[设备] {DEVICE}")

# ==================== GPU确定性设置 (解决GPU训练噪声问题) ====================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 关键: 禁用cudnn非确定算法, 保证可复现且减少噪声
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 不要自动选最优算法(可能不稳定)
    print(f"[GPU稳定性] 已启用deterministic模式, benchmark=False")
    # 检查半精度支持但强制用float32避免精度损失
    if hasattr(torch.cuda, 'amp') and DEVICE.type == 'cuda':
        print(f"[精度] 使用float32训练 (避免fp16噪声)")

# ==================== 热力图生成 ====================
def landmarks_to_heatmap(landmarks_468x3, size=HEATMAP_SIZE, sigma=GAUSS_SIGMA):
    """
    将468个3D关键点转换为热力图
    输入: landmarks (468, 3) 归一化坐标 [0,1]
    输出: heatmap (size, size) float32 [0,1]
    
    多通道版本:
      channel 0: xy平面位置热力图 (所有点的xy投影)
      channel 1: z深度信息 (z值映射到强度)  
      channel 2: 区域分组热力图 (眉毛/眼睛/嘴巴/鼻子 分开画)
    """
    lm = np.asarray(landmarks_468x3, dtype=np.float64)
    if lm.shape[0] != 468:
        return np.zeros((3, size, size), dtype=np.float32)
    
    x = lm[:, 0] * (size - 1)   # 归一化→像素坐标
    y = lm[:, 1] * (size - 1)
    z = lm[:, 2]                # z值 [-1, 1] 左右
    
    # 创建网格坐标
    xx = np.arange(size, dtype=np.float64)
    yy = np.arange(size, dtype=np.float64)
    gx, gy = np.meshgrid(xx, yy)
    
    sigma_px = sigma * size / 10  # sigma转像素
    
    # Channel 0: 全部点的xy位置热力图
    h0 = np.zeros((size, size), dtype=np.float64)
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            h0 += np.exp(-d2 / (2 * sigma_px**2))
    
    # Channel 1: z值加权热力图(z>0的面部凸出部分更亮)
    h1 = np.zeros((size, size), dtype=np.float64)
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            weight = (z[i] + 1) / 2  # 归一化到[0,1]
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            h1 += weight * np.exp(-d2 / (2 * sigma_px**2))
    
    # Channel 2: 关键区域分组 (只画最重要的区域，不同组用不同权重)
    h2 = np.zeros((size, size), dtype=np.float64)
    
    # 关键面部区域分组 (基于mediapipe索引)
    REGIONS = {
        'eyebrow': list(range(70,108)) + list(range(300,338)),     # 眉毛
        'eyes':    [33,133,159,145,386,374,263,362] +              # 眼角+眼睑
                  list(range(33,134)) + list(range(362,394)),       # 眼睛轮廓
        'nose':    [1,2,98,327,168,6,197,195,5],                    # 鼻子关键点
        'mouth':   list(range(61,92)) + list(range(291,322)) +      # 嘴唇外轮廓+内轮廓
                  [13,14,17,0,37,84,87,178,409,375,409,270],        # 嘴唇中心线
        'contour':[10,109,67,103,54,21,162,127,237,137,452,356,345,# 面部轮廓
                  132,172,148,152,377,400,378,382,381,380,361],
    }
    
    region_weights = {'eyebrow': 2.0, 'eyes': 2.5, 'nose': 1.0, 'mouth': 3.0, 'contour': 0.8}
    for region_name, indices in REGIONS.items():
        w = region_weights.get(region_name, 1.0)
        for i in indices:
            if i >= 468: continue
            if 0 <= x[i] < size and 0 <= y[i] < size:
                d2 = (gx - x[i])**2 + (gy - y[i])**2
                h2 += w * np.exp(-d2 / (2 * sigma_px**2))
    
    # 归一化每个channel到[0,1]
    def norm(h):
        mx = h.max()
        return (h / mx if mx > 0 else h).astype(np.float32)
    
    heatmap = np.stack([norm(h0), norm(h1), norm(h2)], axis=0)
    return heatmap


# 向量化加速版 (用于批量生成)
def batch_landmarks_to_heatmap(landmarks_batch, size=HEATMAP_SIZE, sigma=GAUSS_SIGMA):
    """批量生成热力图 (numpy向量化加速)"""
    n = len(landmarks_batch)
    heatmaps = np.zeros((n, 3, size, size), dtype=np.float32)
    
    for i in range(n):
        heatmaps[i] = landmarks_to_heatmap(landmarks_batch[i], size, sigma)
    
    return heatmaps


# ==================== 数据集类 ====================
class FaceEmotionHeatmapDataset(Dataset):
    """从已有推理结果加载landmark→热力图的数据集"""
    
    def __init__(self, data_path, transform=None, size=HEATMAP_SIZE):
        """
        data_path: 包含所有样本的npy文件路径 (由prepare_training_data生成)
        """
        print(f"[数据集] 加载 {data_path} ...")
        data = np.load(data_path, allow_pickle=True).item()
        
        self.heatmaps = data['heatmaps']      # (N, 3, H, W)
        self.labels = data['labels']          # (N,) int
        self.features = data.get('features', None)  # (N, 14) 可选
        self.transform = transform
        
        print(f"[数据集] 共{len(self.heatmaps)}张热力图, {len(np.unique(self.labels))}类")
        class_counts = np.bincount(self.labels)
        for idx, c in enumerate(class_counts):
            if c > 0:
                print(f"  {LABEL_ORDER[idx]}: {c}张")
    
    def __len__(self):
        return len(self.heatmaps)
    
    def __getitem__(self, idx):
        hmap = self.heatmaps[idx].copy()  # (3, H, W)
        label = int(self.labels[idx])
        
        # 数据增强 (V10: 更强增强版抗过拟合)
        if self.transform and np.random.rand() > 0.2:  # 80%做增强(原70%)
            # 随机水平翻转 (左右脸对称)
            if np.random.rand() > 0.5:
                hmap = np.flip(hmap, axis=2).copy()
            
            # 随机旋转 (-20° ~ +20°) — V10: 扩大范围
            if np.random.rand() > 0.4:  # V10: 60%概率旋转(原50%)
                angle = np.random.uniform(-20, 20) * np.pi / 180  # V10: ±20°(原±15°)
                ca, sa = np.cos(angle), np.sin(angle)
                rot = np.array([[ca, -sa], [sa, ca]])
                new_hmap = np.zeros_like(hmap)
                for c in range(3):
                    from scipy.ndimage import affine_transform as aff_t
                    new_hmap[c] = aff_t(
                        hmap[c], rot.T, order=1,
                        output_shape=hmap[c].shape,
                        mode='constant', cval=0
                    )
                hmap = new_hmap
            
            # 随机亮度/对比度/噪声
            if np.random.rand() > 0.5:
                hmap = hmap * np.random.uniform(0.85, 1.15)
                hmap = np.clip(hmap, 0, 1).astype(np.float32)
            
            # 高斯噪声 (V10: 提升概率抗过拟合)
            if np.random.rand() > 0.5:  # V10: 50%加噪声(原30%)
                hmap += np.random.normal(0, 0.02, hmap.shape).astype(np.float32)
                hmap = np.clip(hmap, 0, 1)
        
        return torch.from_numpy(hmap), torch.tensor(label, dtype=torch.long)


class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    解决类别不平衡 + 让模型关注难分类样本"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none',
                                          label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        if alpha is not None:
            at = alpha[targets]
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce
        return loss.mean()


# ==================== MixUp数据增强 ====================
def mixup_data(x, y, alpha=0.2):
    """Mixup: 线性插值两个样本及其标签"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== 轻量CNN模型 ====================
class LightEmotionCNN(nn.Module):
    """
    轻量级CNN: 约5万参数, 推理<1ms(CPU)
    架构: 3层Conv(递增通道) → GlobalAvgPool → FC → 7分类
    
    设计原则:
    - 输入小(64×64×3): 计算量低
    - BatchNorm: 加速收敛+正则化
    - Dropout: 防过拟合
    - GlobalAvgPool: 减少参数 vs Flatten
    """
    
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super().__init__()
        
        # Block 1: 3→16 channels, 提取边缘/点位置
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 64→32
            nn.Dropout2d(dropout_rate*0.5),
        )
        
        # Block 2: 16→32 channels, 局部模式
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 32→16
            nn.Dropout2d(dropout_rate*0.5),
        )
        
        # Block 3: 32→64 channels, 高层语义
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),   # → 4×4
            nn.Dropout2d(dropout_rate),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 64*4*4 = 1024
            nn.Linear(64 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )
    
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# 更深的ResNet-style版本 (可选)
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.conv(x))


class DeepEmotionCNN(nn.Module):
    """
    深度残差CNN: 约12万参数, 更强的表达能力
    用于如果LightCNN效果不够时的升级选项
    
    V9.5: SGD+Nesterov版 (保持原始结构)
    """
    
    def __init__(self, num_classes=7, dropout_rate=0.6):  # V12: dropout 0.5→0.6 (板子数据噪声大需更强正则)
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64→32
        )
        
        # Stage 1: 24→32
        self.stage1 = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 32→16
        )
        
        # Stage 2: 32→48
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48),
            nn.MaxPool2d(2),  # 16→8
        )
        
        # Stage 3: 48→64
        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d(4),  # →4×4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate*0.5),
            nn.Linear(128, num_classes),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ==================== 训练流程 ====================
# ==================== 热力图缓存目录 (v9.2 增量更新版) ====================
# 每张jpg对应一个独立npy文件:  reports/heatmap_cache/{类别}/{原文件名}.npy
# 增删图片只需操作单个文件, 无需全量重生成
HEATMAP_CACHE_DIR = os.path.join(BASE, 'reports', 'heatmap_cache')


def _get_heatmap_path(jpg_path):
    """根据原图路径生成热力图缓存路径"""
    rel = os.path.relpath(jpg_path, BASE)
    cache_path = os.path.join(HEATMAP_CACHE_DIR, rel + '.npy')
    return cache_path


def _load_cached_heatmap(jpg_path):
    """加载单张热力图缓存, 返回(heatmap, features)或(None,None)"""
    cache_path = _get_heatmap_path(jpg_path)
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data['heatmap'], data.get('features')
        except:
            return None, None
    return None, None


def _save_cached_heatmap(jpg_path, hmap, features=None):
    """保存单张热力图缓存"""
    cache_path = _get_heatmap_path(jpg_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, {'heatmap': hmap, 'features': features})


def prepare_training_data(max_per_class=0, save_path=None):
    """
    从原始图片提取landmarks并转为热力图 (v9.2 增量更新版)
    
    增量机制:
      - 每张jpg生成独立缓存: reports/heatmap_cache/{类别}/{文件名}.npy
      - 已存在的缓存直接跳过, 只处理新增图片
      - 被删除的jpg自动排除
    
    返回: dict with 'heatmaps', 'labels', 'features'
    """
    from run_face_keypoint import FaceEmotionDetector
    
    print("=" * 60)
    print("V9.2 准备CNN训练数据 — 提取landmarks → 热力图 (增量模式)")
    print("=" * 60)
    
    det = FaceEmotionDetector(runtime='GPU', log_level='ERROR')
    
    all_hmaps = []
    all_labels = []
    all_features = []
    all_paths = []
    
    total = 0
    cached_count = 0
    new_count = 0
    
    for cfolder, true_label in CATEGORIES.items():
        fdir = os.path.join(BASE, cfolder)
        if not os.path.exists(fdir):
            continue
        files = sorted(glob.glob(os.path.join(fdir, '*.jpg')))
        if max_per_class > 0:
            files = files[:max_per_class]
        
        print(f"  处理 {cfolder}: {len(files)}张 ...", end=" ", flush=True)
        
        t0 = time.perf_counter()
        count = 0
        for fp in files:
            # v9.2: 先查缓存
            hmap, feats = _load_cached_heatmap(fp)
            if hmap is not None:
                # 缓存命中, 直接使用
                all_hmaps.append(hmap)
                all_labels.append(LABEL2IDX[true_label])
                all_paths.append(fp)
                if feats is not None and len(feats) == 14:
                    all_features.append(feats)
                count += 1
                cached_count += 1
                continue
            
            # 缓存未命中, 需要推理
            try:
                r = det.detect(fp)
                if r.get('error'):
                    continue
                
                lm = r['landmarks']
                if lm.shape[0] != 468:
                    continue
                
                hmap = landmarks_to_heatmap(lm)
                
                # 保存到缓存
                feat_list = list(r.get('features', {}).values())
                feats_arr = np.array(feat_list, dtype=np.float32) if len(feat_list) == 14 else None
                _save_cached_heatmap(fp, hmap, feats_arr)
                
                all_hmaps.append(hmap)
                all_labels.append(LABEL2IDX[true_label])
                all_paths.append(fp)
                if feats_arr is not None:
                    all_features.append(feat_list)
                
                count += 1
                new_count += 1
            except Exception as e:
                pass
        
        elapsed = time.perf_counter() - t0
        print(f"✓ {count}张 ({elapsed:.1f}s)")
        total += count
    
    det.release()
    
    heatmaps = np.stack(all_hmaps) if all_hmaps else np.zeros((0, 3, HEATMAP_SIZE, HEATMAP_SIZE))
    labels = np.array(all_labels, dtype=np.int64)
    features = np.array(all_features, dtype=np.float32) if all_features else None
    
    result = {
        'heatmaps': heatmaps,
        'labels': labels,
        'features': features,
        'file_paths': all_paths,
        'label_order': LABEL_ORDER,
        'categories': CATEGORIES,
        'total_samples': total,
        'heatmap_size': HEATMAP_SIZE,
    }
    
    if save_path:
        np.save(save_path, result)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n[保存汇总] {save_path} ({size_mb:.1f}MB)")
    
    # 统计缓存
    cache_files = glob.glob(os.path.join(HEATMAP_CACHE_DIR, '**/*.npy'), recursive=True)
    cache_size = sum(os.path.getsize(f) for f in cache_files) / (1024 * 1024)
    
    print(f"\n总计: {total}张热力图 (缓存命中:{cached_count}, 新增:{new_count})")
    print(f"      形状={heatmaps.shape}")
    print(f"      缓存目录: {HEATMAP_CACHE_DIR} ({len(cache_files)}个文件, {cache_size:.0f}MB)")
    print(f"\n💡 增量提示: 删除/新增图片后再次运行此函数即可自动同步")
    return result


def train_cnn(data_path=None, model_type='light', epochs=EPOCHS, 
              batch_size=BATCH_SIZE, lr=LR, save_model=True):
    """
    训练CNN热力图分类器
    
    Args:
        data_path: 训练数据npy路径 (None则先准备)
        model_type: 'light'(~50K参数) 或 'deep'(~120K参数, 残差网络)
    """
    
    # 准备数据
    cache_dir = os.path.join(BASE, 'reports')
    os.makedirs(cache_dir, exist_ok=True)
    
    if data_path is None or not os.path.exists(data_path):
        # PC模式: 直接使用预生成的128x128数据
        data_path = os.path.join(cache_dir, 'cnn_train_data_128.npy')
        if not os.path.exists(data_path):
            # 尝试旧路径
            data_path = os.path.join(cache_dir, 'cnn_train_data.npy')
            if not os.path.exists(data_path):
                print("[ERROR] 找不到训练数据文件!")
                print("  请先运行: py -3.11 batch_generate_heatmaps.py")
                return None, 0, {}
    
    dataset = FaceEmotionHeatmapDataset(data_path)
    n = len(dataset)
    labels = dataset.labels
    
    # 分层抽样 train/val (8:2)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(n), labels))
    
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    
    # 过采样弱类别 (解决类别不平衡)
    train_labels = [dataset[i][1].item() for i in train_idx]
    
    # 计算每个类的权重 (逆频率)
    class_counts = np.bincount(train_labels, minlength=7)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = np.array([class_weights[l] for l in train_labels])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_idx),
        replacement=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, 
                               num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=False)
    
    print(f"\n[训练集] {len(train_idx)}张 (过采样后每batch约等权)")
    print(f"[验证集] {len(val_idx)}张")
    print(f"[类别分布] {dict(zip(LABEL_ORDER, class_counts.astype(int)))}")
    print(f"[类别权重] {np.round(class_weights/class_weights.sum()*7, 2)}")
    
    # 创建模型
    if model_type == 'light':
        model = LightEmotionCNN(num_classes=7).to(DEVICE)
    else:
        model = DeepEmotionCNN(num_classes=7).to(DEVICE)
    
    params = model.get_num_params()
    print(f"\n[模型] {model_type.upper()} | 参数量={params:,}")
    
    # V12: FocalLoss (解决类别不平衡 + 难样本挖掘)
    cw_array = class_weights / class_weights.sum() * 7  # 归一化权重
    criterion = FocalLoss(alpha=cw_array, gamma=2.0, label_smoothing=0.08)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # V12: 降低max_lr (板子数据噪声大, 高LR过拟合到噪声) + 更长warmup
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*5,
                                              epochs=epochs, steps_per_epoch=len(train_loader),
                                              pct_start=0.15, anneal_strategy='cos')
    
    best_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'Epoch':<6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10} {'LR':>10}")
    print("-" * 62)
    
    for epoch in range(epochs):
        # ---- 训炼 ----
        model.train()
        running_loss = 0.0
        correct = 0
        total_samp = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            batch_size = inputs.size(0)
            
            # V12: MixUp数据增强 (50%概率)
            use_mixup = np.random.rand() > 0.5
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            
            # MixUp时用混合后的准确率估计
            if use_mixup:
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(targets_a).sum().item() +
                           (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR必须每个batch step一次!
            
            total_samp += batch_size
            running_loss += loss.item() * batch_size
        
        train_loss = running_loss / total_samp
        train_acc = correct / total_samp * 100
        
        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        per_class_correct = np.zeros(7)
        per_class_total = np.zeros(7)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += inputs.size(0)
                
                for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                    per_class_total[t] += 1
                    if t == p:
                        per_class_correct[t] += 1
        
        val_loss /= val_total
        val_acc = val_correct / val_total * 100
        
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<6} {train_loss:>10.4f} {train_acc:>9.1f}% "
                  f"{val_loss:>10.4f} {val_acc:>9.1f}% {current_lr:>10.6f}")
        
        # 保存最佳模型 (带版本号, 不覆盖!)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            if save_model:
                acc_str = f"{best_acc:.1f}".replace('.','')
                save_p = os.path.join(cache_dir, f'v15_boardSDK_{model_type}_acc{acc_str}_ep{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'model_type': model_type,
                    'label_order': LABEL_ORDER,
                    'heatmap_size': HEATMAP_SIZE,
                    'data_source': 'board_sdk_fiboaisdk_9906',
                    'train_date': time.strftime('%Y-%m-%d_%H%M'),
                }, save_p)
                # 同时更新一个"latest"软链接(复制)方便引用
                latest_p = os.path.join(cache_dir, f'v15_cnn_{model_type}_latest_best.pth')
                import shutil
                shutil.copy2(save_p, latest_p)
    
    # 最终逐类准确率
    print(f"\n{'─'*60}")
    print(f"最佳验证准确率: {best_acc:.1f}% (Epoch {best_epoch})")
    print(f"\n{'类别':<8} {'准确率':>8} {'样本数':>8}")
    for i in range(7):
        acc_i = per_class_correct[i]/per_class_total[i]*100 if per_class_total[i]>0 else 0
        print(f"{LABEL_ORDER[i]:<8} {acc_i:>7.1f}% {int(per_class_total[i]):>8}")
    
    # 加载最佳模型做最终评估 (使用latest_best)
    save_p = os.path.join(cache_dir, f'v15_cnn_{model_type}_latest_best.pth')
    if not os.path.exists(save_p):
        # fallback: 旧命名
        save_p = os.path.join(cache_dir, f'v9_cnn_{model_type}_best.pth')
    checkpoint = torch.load(save_p, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 详细混淆矩阵分析
    print(f"\n详细评估 (最佳模型):")
    confusion = np.zeros((7, 7), dtype=int)
    all_preds = []
    all_trues = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            for t, p, pb in zip(targets.numpy(), preds.cpu().numpy(), probs.cpu().numpy()):
                confusion[t][p] += 1
                all_trues.append(t)
                all_preds.append(p)
                all_probs.append(pb)
    
    print(f"\n混淆矩阵 (行=真实, 列=预测):")
    header = "         " + "  ".join([f'{e[:3]}>' for e in LABEL_ORDER])
    print(header)
    for i in range(7):
        row = f"  {LABEL_ORDER[i]:>5}  " + "  ".join([f'{confusion[i][j]:>3}' for j in range(7)])
        print(row)
    
    # 与规则系统/SVM对比
    print(f"\n{'─'*60}")
    print("对比总结:")
    print(f"  规则v8.1 精确匹配: ~27% | 复合命中: ~63%")
    print(f"  SVM全量CV:        54.5%")
    print(f"  ★CNN-{model_type.upper()} ValAcc: {best_acc:.1f}% (单标签精确匹配)")
    
    # 保存完整报告
    report = {
        'model_type': model_type,
        'best_val_accuracy': best_acc,
        'best_epoch': best_epoch,
        'params_count': params,
        'history': history,
        'per_class_acc': {LABEL_ORDER[i]: round(per_class_correct[i]/per_class_total[i]*100,1) if per_class_total[i]>0 else 0 for i in range(7)},
        'confusion_matrix': confusion.tolist(),
        'config': {
            'heatmap_size': HEATMAP_SIZE,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'model_type': model_type,
            'device': str(DEVICE),
        },
    }
    
    report_path = os.path.join(cache_dir, f'v15_cnn_{model_type}_report_boardSDK.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")
    
    return model, best_acc, report


# ==================== 主入口 ====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V9.2 CNN热力图情绪分类器 (增量缓存版)')
    parser.add_argument('--prepare-only', action='store_true', help='只准备数据不训练')
    parser.add_argument('--sync', action='store_true', help='增量同步: 只处理新增/删除的图片 (推荐)')
    parser.add_argument('--model-type', default='deep', choices=['light', 'deep'], 
                        help='模型类型: light(~50K参数) 或 deep(~120K参数, 推荐)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--max-per-class', type=int, default=0, help='限制每类数量(0=全部)')
    parser.add_argument('--data-path', type=str, default=None, help='指定训练数据npy路径(默认自动查找)')
    args = parser.parse_args()
    
    # 数据路径: 命令行指定 > 自动查找
    if args.data_path and os.path.exists(args.data_path):
        data_npy = args.data_path
    else:
        data_npy = os.path.join(BASE,'reports','cnn_train_data_128.npy')  # 默认: 128x128高分辨率数据
    
    if args.sync or args.prepare_only:
        print("[PC模式] 跳过板端prepare_training_data, 请使用 batch_generate_heatmaps.py")
        print(f"[PC模式] 数据路径: {data_npy}")
        if not os.path.exists(data_npy):
            print(f"[ERROR] 数据文件不存在! 请先运行: py -3.11 batch_generate_heatmaps.py")
            exit(1)
        exit(0)
    else:
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size
        LR = args.lr
        
        model, acc, report = train_cnn(
            data_path=data_npy if os.path.exists(data_npy) else None,
            model_type=args.model_type,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
        )
