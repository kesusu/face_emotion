# Face Emotion Recognition — 人脸情绪识别
> 基于热力图 + Deep CNN 的7类面部表情识别系统（嵌赛项目）

---

## 项目简介

使用 MediaPipe 提取 468 个面部关键点，生成 3 通道高斯热力图 (128×128)，通过 CNN 进行 7 类情绪分类。

**7 类情绪：** 惊讶 | 恐惧 | 厌恶 | 快乐 | 悲伤 | 愤怒 | 中性

---

## 硬件环境

| 设备 | 用途 |
|---|---|
| **RK3588 开发板** | 板端部署、推理 |
| **PC (RTX 4060 Laptop 8GB)** | 数据处理、模型训练 |

ADB 连接: `adb -s 28de40d2`

---

## 目录结构

```
face_emotion/
├── docs/                          # 所有文档
│   ├── 00_文档索引.md             # 文档导航入口
│   ├── 01_AI项目交接文档_给开发板.md
│   ├── 02_AI优化策略与提示词.md
│   ├── 03_AI接手交接说明_给电脑.md
│   ├── 04_AI_CNN训练报告_PC端.md
│   ├── 05_板端接手交接文档.md
│   ├── 06_全局开发总结与突破方向.md
│   ├── 07_PC替代板端热力图验证方案.md
│   ├── 08_板端协同分工任务单.md
│   ├── 09_PC端总结与路线图.md
│   ├── 10_数据污染修复报告_给板子.md
│   └── todo.md                    # 当前进度/待办事项
│
├── RAF-DB/train/                  # ⭐ 训练数据（干净）
│   ├── 0惊讶/                     # 1575张 ✅ 已修复
│   ├── 1恐惧/                     # 332张
│   ├── 2厌恶/                     # 862张
│   ├── 3快乐/                     # 1108张
│   ├── 4悲伤/                     # 2141张
│   ├── 5愤怒/                     # 814张
│   ├── 6中性/                     # 3074张
│   └── reports/
│       ├── heatmap_cache_128/     # 单张热力图缓存 (~8603个)
│       └── cnn_train_data_128.npy # 最终训练数据集
│
├── train/                         # 数据副本（部分清理过，未完全同步）
├── photos/                        # 板子原始数据备份
├── heatmap_verify/                # 热力图验证样本
│
├── cnn_heatmap_classifier.py      # CNN训练脚本 (Deep CNN)
├── batch_generate_heatmaps.py     # 批量热力图生成器 (MediaPipe, 128x128)
├── heatmap_verify.py              # 热力图验证工具
├── extract_features.py            # 特征提取工具
├── _check_mix.py                  # 数据交叉检查工具
│
├── _noface_0surprise.txt          # 0惊讶中145张无人脸图片列表
└── README.md                      # 本文件
```

---

## 核心流程

```
RAF-DB/train/*.jpg  →  MediaPipe(468关键点)  →  高斯热力图(3ch,128x128)  →  CNN训练  →  .pth模型
```

### 热力图参数
- **尺寸**: 128×128
- **Sigma**: 1.5
- **通道**: Ch0=xy位置高斯 / Ch1=z深度加权 / Ch2=区域分组加权
- **dtype**: float32
- **来源**: PC_MediaPipe_FaceLandmarker

---

## 快速操作

```bash
# 1. 生成/增量更新热力图（已有缓存自动跳过）
py -3.11 batch_generate_heatmaps.py

# 2. 训练CNN模型
py -3.11 cnn_heatmap_classifier.py --model-type deep --epochs 120 --batch-size 64 --lr 0.002

# 3. 推送文件到板子
adb -s 28de40d2 push <本地文件> "/home/fibo/AI model/cv_models/pose/<目标路径>"
```

---

## 数据集状态 (2026-05-09 更新)

| 类别 | 图片数 | 有效热力图 |
|------|:-----:|:---------:|
| 0惊讶 | 1575 | 1430 |
| 1恐惧 | 332 | 288 |
| 2厌恶 | 862 | 754 |
| 3快乐 | 1108 | 980 |
| 4悲伤 | 2141 | 1893 |
| 5愤怒 | 814 | 623 |
| 6中性 | 3074 | 2635 |
| **总计** | **9906** | **~8603** |

### 重要历史事件
- **2026-05-08/09**: 发现并修复 0惊讶类别数据污染（1672张类别4错放文件已清除），详见 `docs/10_数据污染修复报告_给板子.md`

---

## 板端目录对照

| PC端路径 | 板端路径 | 说明 |
|----------|----------|------|
| `RAF-DB/train/` | `photos/train2/` | ✅ 干净数据 |
| `RAF-DB/train/` | `photos/train/` | ⚠️ train/0惊讶 含1683个错放文件，不可用 |

---

## Python 环境
- **推荐版本**: Python 3.11 (`py -3.11`)
- **核心依赖**: numpy, opencv-python, mediapipe, torch/torchvision
