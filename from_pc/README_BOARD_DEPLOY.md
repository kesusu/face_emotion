# CNN情绪模型 — 板子端对接报告

> **版本**: V11-DeepCNN-BoardDeploy  
> **日期**: 2026-05-09 18:12  
> **状态**: ✅ 可部署  

---

## 一、模型概览

| 项目 | 值 |
|------|-----|
| **模型名称** | DeepEmotionCNN (残差深度网络) |
| **参数量** | 480,743 (~48万) |
| **模型大小** | 5.57 MB (`v9_cnn_deep_best.pth`) |
| **输入尺寸** | (3, 128, 128) 三通道热力图, float32, 值域[0,1] |
| **输出** | 7类情绪概率分布 |
| **训练数据** | 板子fiboaisdk生成的9906张热力图 (RAF-DB数据集) |
| **最佳验证准确率** | **37.8%** (Epoch 99) |
| **训练框架** | PyTorch + AdamW + OneCycleLR + FocalLoss |

### 7类标签定义

| 索引 | 中文名 | 训练集样本数 | 验证集准确率 |
|------|--------|-------------|-------------|
| 0 | 惊讶 | 1575 | 26.0% |
| 1 | 恐惧 | 332 | 33.3% |
| 2 | 厌恶 | 862 | 58.3% |
| 3 | 开心 | 1108 | **68.4%** ★最好 |
| 4 | 悲伤 | 2141 | 20.6% ★最差 |
| 5 | 愤怒 | 814 | 60.0% |
| 6 | 中性 | 3074 | 31.7% |

---

## 二、部署包内容清单

```
from_pc/
├── v9_cnn_deep_best.pth          ← 模型权重文件 (5.57MB)
├── cnn_inference_board.py        ← 推理脚本 (板子端运行)
└── README_BOARD_DEPLOY.md        ← 本报告
```

**ADB推送到板子的命令:**
```bash
adb push from_pc/ /data/face_emotion/from_pc/
```

---

## 三、板子端使用方法

### 前置条件
```bash
# PyTorch CPU版 (板子上用CPU推理即可, 5.57MB模型很小)
pip install torch numpy scipy
```

### 方式一: 命令行单张推理
```bash
cd /data/face_emotion/from_pc/

# 对单张热力图推理
python cnn_inference_board.py \
    -m v9_cnn_deep_best.pth \
    -i /path/to/heatmap.npy
```

输出示例:
```
★ 预测结果: 开心 (类别3)
  置信度: 72.3%
──────────────────────────────────────
各类概率:
  开心(3): 72.3% ███████████████████████████
  惊讶(0): 12.1% █████
  厌恶(2): 8.4% ███
  中性(6): 3.2% █
  悲伤(4): 1.8%
  愤怒(5): 1.3%
  恐惧(1): 0.9%
```

### 方式二: Python代码调用 (推荐集成到你的系统)

```python
from cnn_inference_board import EmotionClassifier

# 初始化 (只加载一次)
clf = EmotionClassifier('v9_cnn_deep_best.pth', device='cpu')

# 你的pipeline:
#   1. 拍照 → fiboaisdk检测人脸+468关键点
#   2. 关键点 → 生成128×128热力图 (3通道, float32, [0,1])
#   3. 热力图 → CNN推理

import numpy as np
heatmap = your_heatmap_generation_function(landmarks_468x3)  # shape=(3,128,128)

result = clf.predict(heatmap)
print(f"预测: {result['name']}, 置信度: {result['confidence']:.1%}")
# result['label'] = 3 (int)
# result['name'] = '开心'
# result['confidence'] = 0.723
# result['probs'] = [0.121, 0.009, ...] (7类概率)
```

### 方式三: 批量推理
```bash
python cnn_inference_board.py \
    -m v9_cnn_deep_best.pth \
    -d /path/to/heatmaps_folder/ \
    --json-out results.json
```

---

## 四、热力图格式要求 (关键!)

CNN的输入必须是**与训练数据完全同分布**的热力图:

| 属性 | 要求 | 说明 |
|------|------|------|
| 形状 | `(3, 128, 128)` | CHW格式, 3通道 |
| 数据类型 | `float32` | 必须 |
| 值域 | `[0.0, 1.0]` | 归一化后的高斯热力图 |
| Channel 0 | 全部468点的xy位置热力图 | 所有关键点的高斯叠加 |
| Channel 1 | z深度加权热力图 | 凸出面更亮(z∈[-1,1]→[0,1]) |
| Channel 2 | 区域分组热力图 | 眉毛/眼睛/鼻子/嘴巴分区域画, 权重不同 |
| 高斯sigma | `1.5 * size / 10 = 19.2px` | 固定值 |

**⚠️ 必须使用fiboaisdk提取的468个关键点来生成热力图!**
PC MediaPipe生成的热力图分布不同, 会导致识别率大幅下降。

### 热力图生成伪代码参考
```python
def landmarks_to_heatmap(lm_468x3, size=128, sigma=1.5):
    """
    lm_468x3: fiboaisdk返回的468×3数组, 坐标已归一化到[0,1]
              lm[i] = (x_norm, y_norm, z_norm)
    返回: heatmap (3, 128, 128) float32 [0,1]
    """
    x = lm[:, 0] * (size - 1)   # 归一化→像素坐标
    y = lm[:, 1] * (size - 1)
    z = lm[:, 2]                # z ∈ [-1, 1]
    
    sigma_px = sigma * size / 10
    
    # Ch0: 全部点xy位置
    h0 = sum(gaussian_at(x[i], y[i], sigma_px) for i in range(468))
    
    # Ch1: z加权 (凸出部分更亮)
    weight = (z + 1) / 2         # [-1,1]→[0,1]
    h1 = sum(weight[i] * gaussian_at(x[i], y[i], sigma_px) for i in range(468))
    
    # Ch2: 分区域 (眉毛×2.0, 眼睛×2.5, 鼻子×1.0, 嘴巴×3.0, 轮廓×0.8)
    h2 = draw_region_heatmap(x, y, sigma_px, regions_weights)
    
    return normalize(stack([h0, h1, h2]))
```
完整实现见PC端 `cnn_heatmap_classifier.py` 中的 `landmarks_to_heatmap()` 函数。

---

## 五、推理性能预期

| 设备 | 推理时间 (单张) | 内存占用 |
|------|---------------|---------|
| RK3588 CPU (板子) | ~15-30 ms | ~50 MB |
| PC CPU (i5/i7) | ~5-10 ms | ~50 MB |
| RTX 4060 GPU | <1 ms | ~100 MB |

模型仅48万参数, 在RK3588上可以实时运行 (~30-60 FPS理论上限)。

---

## 六、已知限制 & 改进方向

### 当前问题
1. **整体ValAcc仅37.8%** — 7分类任务在当前数据下较困难
2. **悲伤(20.6%)和惊讶(26%)识别差** — 这两类在板子SDK热力图上区分度低
3. **中性(31.7%)样本最多但难认** — "无表情"本身特征不明显

### 已尝试但效果有限的方案
- FocalLoss + MixUp + Dropout↑ → 过拟合降低(46%→26%), 但天花板只有30%
- 数据增强增强 → 无显著提升
- 调整LR调度器 → 微小波动

### 下一步计划 (待执行)
- **方案A**: 类别合并 7→5类 (恐惧+愤怒→"激动", 悲伤+中性→"低落"), 预计提升到45%+
- **方案B**: PC MediaPipe数据预训→板子数据微调 (域适配)
- **方案C**: 收集更多板子SDK标注数据, 特别是弱类别(恐惧仅332张)

---

## 七、实验记录索引

| 文件 | 内容 |
|------|------|
| `cnn_train_log.txt` | V11完整训练日志 (200epoch逐轮记录) |
| `v9_cnn_report.json` | 结构化训练报告 (per-class混淆矩阵等) |
| `docs/13_CNN训练经验完整总结_V11.md` | V11完整经验总结文档 |

---

## 八、故障排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| `RuntimeError: shape mismatch` | 输入热力图不是(3,128,128) | 检查shape并resize |
| 预测全是一个类别 | 模型加载失败/热力图全黑 | 打印热力图max/min确认[0,1]范围 |
| `KeyError: model_state_dict` | pth文件损坏 | 重新从PC拉取 |
| 识别率明显低于37.8% | 使用了非fiboaisdk的热力图 | 必须用板子SDK的关键点生成 |

---

_报告生成: AI Assistant | 如有问题请检查模型路径和热力图格式_
