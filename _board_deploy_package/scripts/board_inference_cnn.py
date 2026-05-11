#!/usr/bin/env python3
"""
板端CNN情绪推理脚本 (RK3588)
============================
用途: 在RK3588开发板上加载训练好的CNN模型, 对单张热力图做实时情绪推理

输入: 128x128 热力图 numpy array, shape=(3, 128, 128), dtype=float32, range[0,1]
输出: 情绪标签(str) + 置信度(dict) + 推理耗时(ms)

使用方法:
    from board_inference_cnn import EmotionInferencer
    infer = EmotionInferencer(model_path='models/v15_boardSDK_deep_acc306_ep171.pth')
    label, probs, ms = infer.predict(heatmap)  # heatmap: (3,128,128)

命令行测试:
    python board_inference_cnn.py --model models/v15_boardSDK_deep_acc306_ep171.pth

作者: AI | 日期: 2026-05-09 | 版本: V1.0
"""

import os, sys, time, json
import numpy as np


# ==================== 尝试导入PyTorch ====================
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch未安装! 本脚本需要PyTorch环境才能运行")
    print("      请在板子上安装: pip install torch (或使用预编译的RK3588版本)")


# ==================== CNN模型定义 (必须与训练时一致!) ====================

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
    深度残差CNN — 与cnn_heatmap_classifier.py中的DeepEmotionCNN完全一致
    参数量: ~480K
    输入: (batch, 3, 128, 128) float32 [0,1]
    输出: (batch, 7) logits
    """
    def __init__(self, num_classes=7, dropout_rate=0.6):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128->64
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 64->32
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48),
            nn.MaxPool2d(2),  # 32->16
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d(4),  # ->4x4
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
        # 权重初始化 (与训练脚本一致)
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


# ==================== 标签定义 ====================
LABEL_ORDER = ['惊讶', '恐惧', '厌恶', '开心', '悲伤', '愤怒', '中性']
LABEL_CN = LABEL_ORDER  # 兼容别名
NUM_CLASSES = 7


# ==================== 推理器类 ====================
class EmotionInferencer:
    """
    CNN情绪推理器
    
    用法:
        infer = EmotionInferencer('models/v15_boardSDK_deep_acc306_ep171.pth')
        
        # 单张推理
        label, confidences, latency_ms = infer.predict(heatmap_3x128x128)
        
        # 批量推理
        labels, all_probs, latencies = infer.predict_batch(heatmaps_Nx3x128x128)
    
    属性:
        .label_order     : list[str] 7个类别名
        .model_info      : dict       模型元数据(数据源/准确率/日期等)
        .device          : str       推理设备('cpu'/'cuda')
    """

    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: .pth模型文件路径
            device: 'cpu'(板子默认) 或 'cuda'(有GPU时)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch未安装! 无法初始化推理器")

        self.model_path = os.path.abspath(model_path)
        self.device = torch.device(device)
        self.label_order = LABEL_ORDER[:]
        self.model_info = {}
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型权重"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        t0 = time.perf_counter()
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # 创建模型结构
        self.model = DeepEmotionCNN(num_classes=NUM_CLASSES).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()  # 关键: 推理模式!

        # 提取元信息
        self.model_info = {
            'best_acc': ckpt.get('best_acc', '?'),
            'epoch': ckpt.get('epoch', '?'),
            'model_type': ckpt.get('model_type', 'deep'),
            'data_source': ckpt.get('data_source', 'unknown'),
            'train_date': ckpt.get('train_date', 'unknown'),
            'heatmap_size': ckpt.get('heatmap_size', 128),
            'label_order': ckpt.get('label_order', LABEL_ORDER),
            'param_count': sum(p.numel() for p in self.model.parameters()),
        }

        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[模型已加载] {os.path.basename(self.model_path)}")
        print(f"  ValAcc={self.model_info['best_acc']}% | Epoch={self.model_info['epoch']}")
        print(f"  数据源={self.model_info['data_source']}")
        print(f"  参数量={self.model_info['param_count']:,}")
        print(f"  设备={self.device} | 加载耗时={load_ms:.1f}ms")

    @torch.no_grad()
    def predict(self, heatmap):
        """
        对单张热力图做情绪推理
        
        Args:
            heatmap: np.ndarray, shape=(3, H, W), dtype=float32, 值域[0,1]
                     H和W必须是128(与训练一致)
        
        Returns:
            label: str           预测的情绪标签 (如'开心')
            confidences: dict    每类的置信度 {'惊讶': 0.12, ...}
            latency_ms: float    推理耗时(毫秒)
        """
        if not isinstance(heatmap, np.ndarray):
            raise TypeError(f"heatmap必须是numpy数组, 得到{type(heatmap)}")

        t0 = time.perf_counter()
        
        # 预处理: 调整shape和类型
        if heatmap.ndim == 2:
            # 单通道? 扩充为3通道
            heatmap = np.stack([heatmap] * 3, axis=0)
        
        h, w = heatmap.shape[-2], heatmap.shape[-1]
        assert h == 128 and w == 128, f"热力图尺寸必须是128x128, 得到({h}x{w})"
        
        tensor = torch.from_numpy(heatmap.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # 前向推理
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = int(probs.argmax())
        pred_label = self.label_order[pred_idx]
        confidences = {self.label_order[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return pred_label, confidences, latency_ms

    @torch.no_grad()
    def predict_batch(self, heatmaps):
        """
        批量推理
        
        Args:
            heatmaps: np.ndarray, shape=(N, 3, 128, 128)
        
        Returns:
            labels: list[str]      N个预测标签
            all_probs: list[dict]   N个置信度字典
            latencies: list[float]  N个推理耗时(ms)
        """
        t0 = time.perf_counter()
        tensor = torch.from_numpy(heatmaps.astype(np.float32)).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        total_ms = (time.perf_counter() - t0) * 1000

        n = len(heatmaps)
        preds = probs.argmax(axis=1)
        labels = [self.label_order[int(p)] for p in preds]
        all_probs = [{self.label_order[i]: round(float(probs[j][i]), 4) for i in range(NUM_CLASSES)} for j in range(n)]
        per_sample_ms = total_ms / max(n, 1)
        
        return labels, all_probs, [per_sample_ms] * n

    def benchmark(self, n_runs=100, warmup=10):
        """基准测试: 连续推理N次统计延迟"""
        dummy = np.zeros((3, 128, 128), dtype=np.float32)
        
        # warmup
        for _ in range(warmup):
            self.predict(dummy)
        
        times = []
        for _ in range(n_runs):
            _, _, ms = self.predict(dummy)
            times.append(ms)
        
        times = np.array(times)
        print(f"\n{'='*50}")
        print(f'  基准测试 ({n_runs}次推理, {warmup}次预热)')
        print(f'{"="*50}')
        print(f'  平均延迟: {times.mean():.2f}ms')
        print(f'  中位数(P50): {np.median(times):.2f}ms')
        print(f'  P95: {np.percentile(times, 95):.2f}ms')
        print(f'  P99: {np.percentile(times, 99):.2f}ms')
        print(f'  最小: {times.min():.2f}ms / 最大: {times.max():.2f}ms')
        print(f'  FPS: {1000/times.mean():.1f}')
        print(f'{"="*50}')
        return {
            'mean_ms': float(times.mean()),
            'p50_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'fps': float(1000 / times.mean()),
        }

    def get_model_summary(self):
        """返回模型完整信息"""
        info = dict(self.model_info)
        info['device'] = str(self.device)
        info['label_order'] = self.label_order
        info['script_version'] = 'V1.0 (2026-05-09)'
        return info


# ==================== 主入口 (命令行测试) ====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='板端CNN情绪推理测试')
    parser.add_argument('--model', '-m', required=True, help='.pth模型文件路径')
    parser.add_argument('--benchmark', '-b', action='store_true', help='运行基准测试')
    parser.add_argument('--runs', type=int, default=100, help='基准测试次数(默认100)')
    args = parser.parse_args()

    print("="*60)
    print("  板端CNN情绪推理脚本 V1.0")
    print("="*60)

    infer = EmotionInferencer(args.model)

    # 测试推理
    dummy = np.random.rand(3, 128, 128).astype(np.float32) * 0.5
    label, probs, ms = infer.predict(dummy)
    print(f'\n[测试推理]')
    print(f'  输入: 随机热力图 shape={dummy.shape} range=[{dummy.min():.2f},{dummy.max():.2f}]')
    print(f'  预测: {label} ({probs[label]:.1%})')
    print(f'  全部置信度:')
    for e in LABEL_ORDER:
        bar = '█' * int(probs[e] * 30)
        print(f'    {e:>4s}: {probs[e]:6.1%} {bar}')

    # 基准测试
    if args.benchmark:
        result = infer.benchmark(args.runs)
