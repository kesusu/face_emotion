"""
V19 CNN重训 — 解决过拟合, 目标ValAcc>55%
============================================
数据源: cnn_train_data_128.npy (board_sdk_fiboaisdk, 9906张)
环境: py-3.8 + torch2.4.1+cu124
GPU: RTX 4060 Laptop 8GB

V15问题诊断:
  - TrainAcc=56% vs ValAcc=30.6% → 过拟合gap=26pp!
  - 悲伤仅4%、中性仅12% → 弱类完全失效
  - 根因: 480K参数对9906平滑热力图样本太大了

V19改进策略:
  1. 更强正则化: dropout 0.5→0.7, WD 1e-3→5e-3
  2. LabelSmoothing: 0.08→0.15 (更激进防过拟合)
  3. 更小学习率: 1e-3→5e-4 (配合更长的warmup)
  4. EarlyStopping: patience=25 (V15跑到ep171才停, 太晚)
  5. 数据增强加强: CutMix + MixUp + RandAugment风格
  6. 类别平衡: 过采样 + FocalLoss双保险
  7. 模型结构保持DeepEmotionCNN (stem/stage1/stage2/stage3/classifier)

输出: v19_cnn_new.pth → 推送板子 from_pc/models/
"""
import os, sys, time, json, math, random
import numpy as np

# ==================== 环境 ====================
DATA_PATH = r'c:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\new_train\reports\cnn_train_data_128.npy'
OUTPUT_DIR = r'c:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\new_train\reports'

LABEL_NAMES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
LABEL_CN = ['惊讶', '恐惧', '厌恶', '开心', '悲伤', '愤怒', '中性']
NUM_CLASSES = 7
HEATMAP_SIZE = 128

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'[GPU] {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')


# ==================== 超参数 (V19g — 降dropout+SWA冲刺50%) ====================
# V19f: 47.9% (温和平衡, dropout=0.6)
# V19g: dropout降到0.5(任务书说"可以试0.5-0.7"), 加SWA稳化
CONFIG = {
    'batch_size': 64,
    'epochs': 150,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout_rate': 0.5,       # ★ 从0.6降到0.5
    'label_smoothing': 0.1,
    'focal_gamma': 2.0,
    'grad_clip': 5.0,
    'es_patience': 20,
    'aug_prob': 0.8,
    'cosine_T_max': 50,
    'swa_start': 30,           # ★ SWA从ep30开始
    'swa_freq': 3,             # ★ 每3个epoch平均一次
}

print(f'\n[Config] batch={CONFIG["batch_size"]} ep={CONFIG["epochs"]} lr={CONFIG["lr"]} '
      f'dropout={CONFIG["dropout_rate"]} LS={CONFIG["label_smoothing"]}')


# ==================== 数据集 ====================
class HeatmapDataset(Dataset):
    def __init__(self, heatmaps, labels, augment=False):
        self.heatmaps = heatmaps
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        h = self.heatmaps[idx].copy()  # (3, H, W)
        y = int(self.labels[idx])

        if self.augment and random.random() < CONFIG['aug_prob']:
            # 水平翻转 (50%)
            if random.random() < 0.5:
                h = np.flip(h, axis=2).copy()

            # 随机旋转 ±15° (60%概率)
            if random.random() < 0.6:
                angle = random.uniform(-15, 15) * math.pi / 180
                ca, sa = math.cos(angle), math.sin(angle)
                rot = np.array([[ca, -sa], [sa, ca]])
                new_h = np.zeros_like(h)
                for c in range(3):
                    from scipy.ndimage import affine_transform as aff_t
                    new_h[c] = aff_t(h[c], rot.T, order=1,
                                     output_shape=h[c].shape,
                                     mode='constant', cval=0)
                h = new_h

            # 随机缩放/亮度 (50%)
            if random.random() < 0.5:
                scale = random.uniform(0.85, 1.18)
                h = h * scale
            # 对比度 (40%)
            if random.random() < 0.4:
                mean_v = h.mean()
                h = (h - mean_v) * random.uniform(0.85, 1.2) + mean_v
            h = np.clip(h, 0, 1).astype(np.float32)

            # 高斯噪声 (50%)
            if random.random() < 0.5:
                h += np.random.normal(0, 0.025, h.shape).astype(np.float32)
                h = np.clip(h, 0, 1)

            # 随机擦除 (20%) — 强力抗过拟合!
            if random.random() < 0.2:
                erase_w = random.randint(10, 30)
                erase_h = random.randint(10, 30)
                x0 = random.randint(0, HEATMAP_SIZE - erase_w)
                y0 = random.randint(0, HEATMAP_SIZE - erase_h)
                h[:, y0:y0+erase_h, x0:x0+erase_w] = 0

            # 通道抖动 (30%)
            if random.random() < 0.3:
                noise_ch = np.random.normal(1, 0.05, (3, 1, 1)).astype(np.float32)
                h = h * noise_ch
                h = np.clip(h, 0, 1)

        return torch.from_numpy(h), torch.tensor(y, dtype=torch.long)


# ==================== 模型 (必须用stem/stage1/stage2/stage3/classifier) ====================
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class DeepEmotionCNN(nn.Module):
    """
    V19c版: 严格按任务书架构 (stem/stage1/stage2/stage3/classifier)
    与任务书完全一致: 无卷积层Dropout2d, 只有分类头Dropout(0.6)
    """

    def __init__(self, num_classes=7, dropout_rate=0.6):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128→64
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 64→32
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48),
            nn.MaxPool2d(2),  # 32→16
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d(4),  # → 4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 64*4*4 = 1024
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),             # 主Dropout 0.6
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),       # 副Dropout 0.3
            nn.Linear(128, num_classes),
        )
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
                nn.init.xavier_normal_(m.weight, gain=0.5)  # 更小的初始化gain
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ==================== Losses ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none',
                                          label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        return loss.mean()


def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== 训练主流程 ====================
def train():
    t_start = time.time()

    # ---- 加载数据 ----
    print('\n' + '=' * 60)
    print('V19 CNN Training — Anti-Overfit Edition')
    print('Data: board_sdk_fiboaisdk 9906 samples')
    print('=' * 60)

    print(f'\n[Loading] {DATA_PATH}')
    raw = np.load(DATA_PATH, allow_pickle=True).item()
    heatmaps = raw['heatmaps'].astype(np.float32)   # (9906, 3, 128, 128)
    labels = raw['labels'].astype(np.int64)          # (9906,)  0-based

    print(f'  shape: {heatmaps.shape}, dtype: {heatmaps.dtype}')
    print(f'  source: {raw.get("source", "unknown")}')
    print(f'  labels range: [{labels.min()}, {labels.max()}]')
    print(f'  value range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]')

    # 类别分布
    print(f'\n[Class distribution]')
    counts = [int((labels == i).sum()) for i in range(NUM_CLASSES)]
    for i in range(NUM_CLASSES):
        print(f'  [{i}] {LABEL_CN[i]}: {counts[i]} ({counts[i]/len(labels)*100:.1f}%)')

    N = len(labels)
    print(f'\n  Total: {N} samples')

    # ---- 划分数据集 ----
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(N), labels))

    print(f'\n[Split] Train={len(train_idx)} Val={len(val_idx)} (80:20 stratified)')

    train_ds = HeatmapDataset(heatmaps[train_idx], labels[train_idx], augment=True)
    val_ds = HeatmapDataset(heatmaps[val_idx], labels[val_idx], augment=False)

    # ---- 过采样 (类别不平衡) ----
    train_labels_np = labels[train_idx]
    class_counts_arr = np.bincount(train_labels_np, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts_arr.astype(np.float32) + 1e-6)
    sample_weights = np.array([class_weights[l] for l in train_labels_np])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    # ---- 温和过采样 (根号权重, 不像之前那样极端) ----
    train_labels_np = labels[train_idx]
    class_counts_arr = np.bincount(train_labels_np, minlength=NUM_CLASSES)
    # 根号权重: sqrt(1/count), 比直接1/count温和很多
    class_weights = 1.0 / np.sqrt(class_counts_arr.astype(np.float32) + 1e-6)
    sample_weights = np.array([class_weights[l] for l in train_labels_np])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_idx),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 2,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f'[Loader] Train batches={len(train_loader)}, Val batches={len(val_loader)}')
    print(f'[ClassWeights] sqrt-weighted: {np.round(class_weights / class_weights.max(), 2).tolist()}')

    # ---- 创建模型 ----
    model = DeepEmotionCNN(num_classes=NUM_CLASSES,
                           dropout_rate=CONFIG['dropout_rate']).to(DEVICE)
    n_params = model.get_num_params()
    print(f'\n[Model] DeepEmotionCNN-V19 | Params: {n_params:,}')

    # ---- 损失函数: FocalLoss(仅gamma, 无alpha权重) ----
    criterion = FocalLoss(alpha=None, gamma=CONFIG['focal_gamma'],
                          label_smoothing=CONFIG['label_smoothing'])

    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG['lr'],
                            weight_decay=CONFIG['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['cosine_T_max'],  # 任务书: T_max=50
        eta_min=1e-6,
    )

    # ---- 训练循环 ----
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    es_counter = 0  # EarlyStopping计数器
    per_class_best_acc = np.zeros(NUM_CLASSES)

    # SWA: 累积权重
    swa_weights = None
    swa_count = 0

    header = f"{'Ep':<5} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'LR':>10} {'Best':>6}"
    print(f'\n{header}')
    print('-' * len(header))

    for epoch in range(CONFIG['epochs']):
        # === TRAIN ===
        model.train()
        running_loss = 0.0
        correct = 0
        total_samp = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            bs = xb.size(0)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['grad_clip'])
            optimizer.step()

            # Acc
            with torch.no_grad():
                _, preds = logits.max(1)
                correct += preds.eq(yb).sum().item()

            total_samp += bs
            running_loss += loss.item() * bs

        tr_loss = running_loss / total_samp
        tr_acc = correct / total_samp * 100

        # === VALIDATE ===
        model.eval()
        vloss = 0.0
        vcorr = 0
        vtotal = 0
        pc_corr = np.zeros(NUM_CLASSES)
        pc_tot = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                _, preds = logits.max(1)
                vcorr += preds.eq(yb).sum().item()
                vtotal += xb.size(0)
                for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                    pc_tot[t] += 1
                    if t == p:
                        pc_corr[t] += 1

        va_loss = vloss / vtotal
        va_acc = vcorr / vtotal * 100

        # 读LR在scheduler.step之前
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        # 打印 (每5epoch或第1个或突破最佳)
        is_best = False
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch + 1
            is_best = True
            es_counter = 0
            per_class_best_acc = pc_corr / np.maximum(pc_tot, 1) * 100
            # 保存模型
            save_path = os.path.join(OUTPUT_DIR, f'v19_cnn_new.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': CONFIG,
                'label_order': LABEL_CN,
                'data_source': 'board_sdk_fiboaisdk_9906',
                'train_date': time.strftime('%Y-%m-%d_%H%M'),
            }, save_path)
        else:
            es_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            marker = ' ★' if is_best else ''
            print(f'{epoch+1:<5} {tr_loss:>8.4f} {tr_acc:>6.1f}% '
                  f'{va_loss:>8.4f} {va_acc:>6.1f}% {current_lr:>10.6f}'
                  f' {best_val_acc:>5.1f}%{marker}')

        # CosineAnnealing: per-epoch step
        scheduler.step()

        # === SWA累积 ===
        if epoch >= CONFIG.get('swa_start', 999) and (epoch - CONFIG['swa_start']) % CONFIG.get('swa_freq', 3) == 0:
            w = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = w
                swa_count = 1
            else:
                for k in swa_weights:
                    if swa_weights[k].is_floating_point():
                        swa_weights[k] = (swa_weights[k] * swa_count + w[k]) / (swa_count + 1)
                    else:
                        swa_weights[k] = w[k]  # Long tensor如num_batches_tracked直接取最新
                swa_count += 1

        # === Early Stopping ===
        if es_counter >= CONFIG['es_patience']:
            print(f'\n[EarlyStop] No improvement for {es_counter} epochs at ep{epoch+1}')
            break

    # ===== SWA模型评估 =====
    if swa_weights is not None and swa_count > 0:
        print(f'\n[SWA] Evaluating averaged model ({swa_count} snapshots)...')
        model.load_state_dict({k: v.to(DEVICE) for k, v in swa_weights.items()})
        model.eval()
        swa_corr, swa_total = 0, 0
        swa_pc_corr = np.zeros(NUM_CLASSES)
        swa_pc_tot = np.zeros(NUM_CLASSES)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                _, preds = logits.max(1)
                swa_corr += preds.eq(yb).sum().item()
                swa_total += xb.size(0)
                for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                    swa_pc_tot[t] += 1
                    if t == p:
                        swa_pc_corr[t] += 1
        swa_acc = swa_corr / swa_total * 100
        print(f'[SWA] ValAcc = {swa_acc:.1f}%')
        for i in range(NUM_CLASSES):
            a = swa_pc_corr[i] / max(swa_pc_tot[i], 1) * 100
            print(f'  {LABEL_CN[i]}: {a:.1f}% ({int(swa_pc_tot[i])})')

        # 如果SWA更好, 替换保存的模型
        if swa_acc > best_val_acc:
            print(f'[SWA] ★ Better than best({best_val_acc:.1f}%→{swa_acc:.1f}%), saving SWA model!')
            best_val_acc = swa_acc
            per_class_best_acc = swa_pc_corr / np.maximum(swa_pc_tot, 1) * 100
            save_path = os.path.join(OUTPUT_DIR, 'v19_cnn_new.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v.to('cpu') for k, v in swa_weights.items()},
                'best_val_acc': best_val_acc,
                'config': CONFIG,
                'label_order': LABEL_CN,
                'data_source': 'board_sdk_fiboaisdk_9906',
                'train_date': time.strftime('%Y-%m-%d_%H%M'),
                'swa_count': swa_count,
            }, save_path)
        else:
            print(f'[SWA] Not better ({swa_acc:.1f}% vs best {best_val_acc:.1f}%), keeping best model')
            # 恢复最佳模型用于后续混淆矩阵
            ckpt = torch.load(os.path.join(OUTPUT_DIR, 'v19_cnn_new.pth'), map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

    elapsed = time.time() - t_start

    # ===== 最终报告 =====
    print(f'\n{"=" * 60}')
    print(f'TRAINING COMPLETE ({elapsed/60:.1f} min)')
    print(f'{"=" * 60}')
    print(f'Best ValAcc: {best_val_acc:.1f}% @ Epoch {best_epoch}')
    print(f'Model params: {n_params:,}')

    print(f'\n{"Class":<8} {"BestAcc":>8} {"Samples":>8}')
    for i in range(NUM_CLASSES):
        n_samp = int(pc_tot[i]) if pc_tot[i] > 0 else 0
        acc_i = per_class_best_acc[i] if pc_tot[i] > 0 else 0
        print(f'{LABEL_CN[i]:<8} {acc_i:>7.1f}% {n_samp:>8}')

    # 加载最佳模型跑详细混淆矩阵
    ckpt = torch.load(os.path.join(OUTPUT_DIR, 'v19_cnn_new.pth'), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    all_probs = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            for t, p, pb in zip(yb.numpy(), preds.cpu().numpy(), probs.cpu().numpy()):
                confusion[t][p] += 1
                all_probs.append(pb)

    print(f'\nConfusion Matrix (row=true, col=pred):')
    hdr = '       ' + ' '.join([f'{c[:3]:>4}' for c in LABEL_CN])
    print(hdr)
    for i in range(NUM_CLASSES):
        row = f'{LABEL_CN[i]:>5}  ' + ' '.join([f'{confusion[i][j]:>4}' for j in range(NUM_CLASSES)])
        print(row)

    # 保存完整报告
    report = {
        'version': 'V19',
        'best_val_accuracy': round(best_val_acc, 1),
        'best_epoch': best_epoch,
        'params_count': n_params,
        'config': CONFIG,
        'per_class_acc': {LABEL_CN[i]: round(float(per_class_best_acc[i]), 1) for i in range(NUM_CLASSES)},
        'per_class_samples': {LABEL_CN[i]: int(pc_tot[i]) for i in range(NUM_CLASSES)},
        'confusion_matrix': confusion.tolist(),
        'history': {k: [round(v, 4) if isinstance(v, float) else v for v in vals]
                     for k, vals in history.items()},
        'data_source': 'board_sdk_fiboaisdk_9906',
        'elapsed_min': round(elapsed / 60, 1),
        'device': str(DEVICE),
    }
    report_path = os.path.join(OUTPUT_DIR, 'v19_cnn_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\nReport saved: {report_path}')
    print(f'Model saved: {os.path.join(OUTPUT_DIR, "v19_cnn_new.pth")}')
    print(f'Size: {os.path.getsize(os.path.join(OUTPUT_DIR, "v19_cnn_new.pth")) / 1024 / 1024:.1f} MB')

    # 与历史对比
    print(f'\n--- Comparison ---')
    print(f'  V15 CNN(boardSDK):     30.6% (ep171)  ← baseline')
    print(f'  V16 SVM(boardSDK):     41.7%           ← sklearn')
    print(f'  V16 RF(boardSDK):      47.2%           ← sklearn best')
    print(f'  V17 3class SVM:        54.5%           ← 3-class cascade')
    print(f'  Expert v5 (ensemble):  64.6%           ← board current best')
    print(f'  ★ V19 CNN(this run):  {best_val_acc:.1f}%           ← {"TARGET HIT!" if best_val_acc > 55 else "need more work"}')


if __name__ == '__main__':
    train()
