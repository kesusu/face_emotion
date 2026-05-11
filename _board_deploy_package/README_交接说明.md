# 板端部署交接包 — README

> **交接时间**: 2026-05-09 20:39  
> **PC端操作者**: AI  
> **目标设备**: RK3588开发板 (ADB: 28de40d2)  
> **状态**: ✅ 已完成推送  

---

## 一、包内容总览

```
pose/训练交接包/from_pc/
│
├── 📁 models/                          ★ 模型文件
│   ├── v15_boardSDK_deep_acc306_ep171.pth   ← 最佳CNN模型 (5.7MB, ValAcc=30.6%)
│   └── v15_cnn_deep_latest_best.pth         ← 最新best副本 (同一模型)
│
├── 📁 scripts/                         ★ 板端运行脚本
│   └── board_inference_cnn.py               ← CNN推理脚本 (含模型定义+API)
│
├── 📁 reports/                          ★ 训练报告
│   └── v15_cnn_deep_report_boardSDK.json    ← 完整训练报告(混淆矩阵/逐类acc/超参)
│
└── 📄 README_交接说明.md                  ← 本文件
```

---

## 二、模型详情

### 2.1 CNN Deep模型 (`v15_boardSDK_deep_acc306_ep171.pth`)

| 项目 | 值 |
|------|-----|
| 架构 | DeepEmotionCNN (残差网络, 3-stage) |
| 参数量 | **480,743** |
| 文件大小 | **5.7 MB** |
| 输入规格 | `(batch, 3, 128, 128)` float32, 值域[0,1] |
| 输出规格 | `(batch, 7)` logits → softmax → 概率 |
| 训练数据源 | **板子SDK fiboaisdk** 9906张热力图 |
| **验证准确率** | **30.6%** (Epoch 171, 1982张验证集) |
| 训练框架 | PyTorch 2.4.1 + AdamW + OneCycleLR + FocalLoss + MixUp |
| 训练设备 | PC RTX 4060 Laptop GPU (CUDA) |

### 2.2 逐类准确率

| 类别 | 准确率 | 验证样本数 | 表现评价 |
|------|--------|-----------|----------|
| 愤怒 | **62.0%** | 163 | ✅ 强 |
| 厌恶 | **60.1%** | 173 | ✅ 强 |
| 开心 | **58.6%** | 222 | ✅ 好 |
| 惊讶 | **51.7%** | 315 | ⚠️ 中等 |
| 恐惧 | **27.3%** | 66 | ❌ 弱(样本少) |
| 中性 | **9.3%** | 615 | ❌❌ 很弱 |
| 悲伤 | **5.6%** | 428 | ❌❌ 很弱 |

> ⚠️ **注意**: CNN在这个任务上的表现不如传统ML模型(RF=47.7%, KNN=45.3%)。  
> 这是因为热力图是平滑高斯图，缺乏CNN擅长的边缘/纹理特征，导致严重过拟合(Train 56% vs Val 30%)。

### 2.3 模型内部字段 (checkpoint keys)

```python
{
    'epoch': 171,                        # 最佳epoch编号
    'model_state_dict': {...},           # 模型权重 (用于model.load_state_dict)
    'optimizer_state_dict': {...},       # 优化器状态 (继续训练用)
    'best_acc': 30.6,                    # 验证集最佳准确率(%)
    'model_type': 'deep',                # 模型变体标识
    'label_order': [...],                # 7个类别名称列表
    'heatmap_size': 128,                 # 训练时热力图尺寸
    'data_source': 'board_sdk_fiboaisdk_9906',  # 数据来源标记
    'train_date': '2026-05-09_XXXX',     # 训练完成时间
}
```

---

## 三、板端使用指南

### 3.1 环境依赖

```bash
# 板子上需要的Python库
pip install torch torchvision numpy
# 注意: RK3588可能需要特定版本的PyTorch (ARM aarch64)
```

### 3.2 快速开始

```python
# 方式1: 作为模块导入使用
from scripts.board_inference_cnn import EmotionInferencer

infer = EmotionInferencer('models/v15_boardSDK_deep_acc306_ep171.pth')

# 对单张热力图推理 (热力图由板子SDK的fiboaisdk生成)
label, probs, ms = infer.predict(my_heatmap)  # my_heatmap: (3,128,128) float32
print(f'预测: {label}, 置信度: {probs[label]}, 耗时: {ms:.1f}ms')

# 方式2: 命令行测试
python scripts/board_inference_cnn.py --model models/v15_boardSDK_deep_acc306_ep171.pth --benchmark
```

### 3.3 与板子SDK对接流程

```
板子摄像头 → fiboaisdk检测人脸 → 468个3D关键点
                                        ↓
                              landmarks_to_heatmap()  (生成128×128热力图)
                                        ↓
                              EmotionInferencer.predict(heatmap)
                                        ↓
                              输出: 情绪标签 + 7类概率 + 延迟
```

### 3.4 推理性能参考值

| 环境 | 平均延迟 | P95 | FPS | 备注 |
|------|---------|-----|-----|------|
| PC i7/Ryzen + CPU | ~20ms | ~25ms | ~50 | 无GPU时 |
| PC RTX 4060 GPU | ~3ms | ~5ms | ~300+ | CUDA加速 |
| **RK3588 ARM CPU (预估)** | **~80-150ms** | ~200ms | **~7-12** | 无SIMD优化 |

> ⚠️ 板子CPU上推理较慢 (~100ms/帧)，如需实时(>10fps)，考虑：
> 1. 使用更轻量的LightCNN模型(~50K参数, 但准确率更低)
> 2. 改用RandomForest模型(47.7%准确率, <10ms推理)
> 3. 降低输入分辨率到64×64

---

## 四、重要注意事项

### 4.1 数据兼容性

- **本模型只能处理板子SDK生成的热力图** (fiboaisdk版本)
- PC MediaPipe生成的热力图**不兼容**(关键点数量/索引不同)
- 热力图尺寸**必须是128×128**, 其他尺寸会报错
- 热力图通道顺序: `[XY位置, Z深度, 区域分组]`

### 4.2 已知局限

1. **中性/悲伤识别极差** (9%/6%)：这两类样本多但特征模糊，CNN几乎全猜错
2. **过拟合严重**: 训练集56% vs 验证集30%，gap=26%
3. **不是最终方案**: RF(47.7%)和Ensemble融合(预期55%)才是正确方向

### 4.3 如果要改进

按优先级排序:
1. **改用RF/SVM模型** (见策略文档 `docs/14_策略反思_从CNN到多模型修正.md`)
2. **Ensemble融合**: RF + SVM(手工特征) + 规则系统三模型投票
3. **5类合并方案**: 合并恐惧+愤怒→"激动", 减少类别提升每类可分性
4. **XGBoost/LightGBM**: 替代RF, 可能再提1-3%

---

## 五、文件版本历史

| 时间 | 操作 | 说明 |
|------|------|------|
| 2026-05-09 19:00 | CNN训练完成 | py3.8 + GPU, 200ep, ValAcc=30.6% |
| 2026-05-09 20:39 | 打包推送 | 整理为部署包, ADB推送到板子 |

---

## 六、联系与反馈

如有问题请检查以下文件获取更多上下文:

- `docs/14_策略反思_从CNN到多模型修正.md` — 完整策略分析(为什么CNN不行)
- `docs/13_CNN训练经验完整总结_V11.md` — CNN调参历史记录
- `RAF-DB/train/reports/v15_cnn_deep_report_boardSDK.json` — 本次训练详细报告
