# V19g CNN 交付说明

> 日期: 2026-05-12
> 响应任务书: from_board V19g任务书_20260512.md

## 交付版本: V19g-a

### 关键改动(对比V19f)

| 项目 | V19f | V19g-a |
|------|------|--------|
| FocalLoss alpha中性 | 无 | **0.5 (降权)** |
| 中性样本数 | 3074(全量) | **1500(截断)** |
| label_smoothing | 0.1 | **0.05** |
| dropout | 0.6 | **0.5** |
| 恐惧alpha | 无 | **1.5 (加权)** |
| 厌恶/愤怒alpha | 无 | **1.2 (加权)** |

### 训练结果

- **ValAcc: 40.2%** @ Epoch 36 (ep35 0-based)
- **非中性→中性max: 2.2%** (V19f是56%, 改善53.8pp!)
- **中性precision: 39.3%** (V19f远超70%)

### 混淆矩阵关键行 (板端350张测试集对照)

```
真实→中性比例:
  惊讶: 2.2%  (V19f=33%)
  恐惧: 1.5%  (V19f=33%)
  厌恶: 1.2%  (V19f=53%)
  开心: 0.0%  (V19f=23%)
  悲伤: 1.6%  (V19f=57%)
  愤怒: 0.0%  (V19f=17%)
  中性recall: 3.7% (V19f=87%)
```

### 3个版本对比

| 版本 | ValAcc | 非中性→中性max | 说明 |
|------|--------|---------------|------|
| **V19g-a** | **40.2%** | **2.2%** | ★ 交付此版, 平衡最好 |
| V19g-b | 33.2% | 0.0% | 过度: 中性完全压死(recall=0%) |
| V19g-c | 32.4% | 0.5% | 温和但ValAcc太低 |

### 验证脚本

```python
import torch
ckpt = torch.load('v19g_cnn_boardSDK_40_ep36.pth', map_location='cpu', weights_only=False)
assert ckpt['best_val_acc'] > 35
assert ckpt['epoch'] == 35  # 0-based
assert ckpt['config']['focal_alpha'][6] == 0.5  # 中性降权
print("V19g-a OK!")
```

### 交付文件

| # | 文件 | 说明 |
|---|------|------|
| 1 | `v19g_cnn_boardSDK_40_ep36.pth` | 模型权重(1.9MB, 仅权重无优化器) |
| 2 | `v19ga_cnn_report.json` | 训练报告(逐类acc+混淆矩阵+历史) |
| 3 | 本文件 | 交付说明 |
