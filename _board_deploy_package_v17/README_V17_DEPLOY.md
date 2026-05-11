# V17 三类专用SVM 部署包 - 板端使用指南

> **版本**: 2026-05-09 22:03  
> **用途**: 解决 厌恶/悲伤/中性 三类混淆问题  
> **PC训练完成，板子直接用**

---

## 一、文件清单

```
from_pc/
├── models/
│   ├── v16_svm_cleaned_56%_20260509_2136.joblib    # 7类基线SVM (13.4MB) ★ 必须有
│   ├── v16_rf_cleaned_56%_20260509_2136.joblib     # 7类RF辅助 (83MB) 可选
│   └── v17_3class_svm_63%_20260509_2158.joblib     # ★ 三类专用SVM (12MB) ★ 核心新模型
│
├── docs/
│   └── 15_两级级联策略_厌恶悲伤中性.md              # 完整策略设计文档
│
└── README_V17_DEPLOY.md                             # 本文件
```

---

## 二、核心问题：厌恶/悲伤/中性分不清

### 当前V16(7类SVM)在三类上的表现：

| 类别 | 准确率 | 主要误判方向 |
|------|--------|------------|
| **厌恶** | ~9-20% | 54%被判成中性 |
| **悲伤** | ~35-45% | 50%被判成中性 |
| **中性** | 87-94% | 识别很好 |

**本质**：不是"三类互混"，而是**厌恶和悲伤全崩向中性**。中性是个黑洞。

### V17解决思路：两级级联

```
输入: 128x128 热力图 (3通道)
       ↓
[第一级] V17 三类专用SVM → 判断是{厌恶,悲伤,中性}中的哪一个
       ↓
[第二级] 板子原有Ensemble(SVM+CNN+Rules) → 7类预测
       ↓
[融合决策]
  IF 第二级输出 ∈ {惊喜,恐惧,快乐,愤怒}
      → 直接用第二级结果 (板子强的类别不动)
  ELSE (第二级输出 ∈ {厌恶,悲伤,中性})
      → 用第一级V17的结果 (专门优化的判断)
       ↓
最终输出
```

### V17效果（验证集945张）：

| 类别 | V16(7类) | **V17(3类专用)** | 提升 |
|------|---------|-----------------|------|
| **厌恶** | 20.0% | **46.2%** | **+26pp** |
| **悲伤** | 44.9% | **54.5%** | **+10pp** |
| **中性** | 93.8% | 72.8% | -21pp |
| **3类总体** | ~57% | **62.9%** | **+6pp** |

> 中性下降可接受 — 用户原话："没到厌恶/悲伤的程度就判中性也行"

---

## 三、模型加载与使用

### 3.1 加载模型

```python
import joblib
import numpy as np

# === 加载V17 (三类专用) ===
pkg_v17 = joblib.load('models/v17_3class_svm_63%_20260509_2158.joblib')

svm_3 = pkg_v17['model']           # sklearn SVC (RBF, C=10, balanced)
pca = pkg_v17['pca']               # PCA(50 components), 和V16共用!
scaler = pkg_v17['scaler']         # StandardScaler, 和V16共用!

label_map = pkg_v17['label_map_internal']   # {0:2, 1:4, 2:6} 内部→原始ID
label_rev = pkg_v17['label_map_rev']        # {2:0, 4:1, 6:2} 原始→内部

# === 加载V16 (7类基线) ===
pkg_v16 = joblib.load('models/v16_svm_cleaned_56%_20260509_2136.joblib')
svm_7 = pkg_v16['model']           # 7类SVM
# pca和scaler和v17完全一样，不需要重复加载
```

### 3.2 单张热力图推理

```python
def predict_cascade(heatmap):
    """
    heatmap: numpy array, shape=(3, 128, 128), float32
              就是你们板子SDK生成的那个热力图
    
    返回: {
        'final_label': int,          # 最终预测的类别ID (0-6)
        'final_name': str,           # 类别名如 "4悲伤"
        'stage1_result': int or None,# V17结果(仅当进入弱类时有效)
        'stage2_result': int,        # V16/板子原有结果
        'confidence': float,         # 置信度
    }
    """
    # Step 1: 展平 → 标准化 → PCA降维 (49150维 → 50维)
    x_flat = heatmap.flatten().reshape(1, -1).astype(np.float32)
    x_scaled = scaler.transform(x_flat)
    x_50 = pca.transform(x_scaled)
    
    # Step 2: 第二级 - 7类基线预测 (板子原有的逻辑)
    pred_7 = svm_7.predict(x_50)[0]          # 0-6
    prob_7 = svm_7.predict_proba(x_50)[0]    # 7类概率
    conf_7 = prob_7.max()
    
    # 强类列表: 惊喜/恐惧/快乐/愤怒 (这些板子上效果好)
    STRONG_CLASSES = {0, 1, 3, 5}
    
    if pred_7 in STRONG_CLASSES:
        # 板子强项 → 直接用
        return {
            'final_label': int(pred_7),
            'final_name': ['0惊讶','1恐惧','2厌恶','3快乐','4悲伤','5愤怒','6中性'][pred_7],
            'stage1_result': None,
            'stage2_result': int(pred_7),
            'confidence': float(conf_7),
            'used_stage': 'stage2_only',
        }
    
    # Step 3: 第一级 - 三类专用SVM (只处理弱类: 厌恶/悲伤/中性)
    pred_3 = svm_3.predict(x_50)[0]           # 0/1/2 (内部标签)
    prob_3 = svm_3.predict_proba(x_50)[0]     # 3类概率
    conf_3 = prob_3.max()
    
    # 映射回原始7类ID
    final_label = label_map[pred_3]           # {0:2, 1:4, 2:6}
    
    # ★ 可选优化: 低置信度降级为中性
    CONF_THRESHOLD = 0.40  # 低于这个置信度就判中性
    if conf_3 < CONF_THRESHOLD:
        final_label = 6  # 降级为中性
    
    return {
        'final_label': final_label,
        'final_name': ['0惊讶','1恐惧','2厌恶','3快乐','4悲伤','5愤怒','6中性'][final_label],
        'stage1_result': int(label_map[pred_3]),
        'stage2_result': int(pred_7),
        'confidence': float(conf_3),
        'used_stage': 'cascade',
    }
```

### 3.3 批量推理示例

```python
# 假设你有一批热力图
heatmaps_batch = [...]  # list of (3,128,128) arrays

results = []
for hm in heatmaps_batch:
    r = predict_cascade(hm)
    results.append(r['final_label'])

print(f"预测结果: {results}")
```

---

## 四、关键细节

### 4.1 PCA/Scaler为什么共享？

V17复用了V16的PCA和Scaler。这意味着：
- **两级的特征空间完全一致**
- 不需要额外维护两套变换器
- 如果以后重训V16，V17必须跟着更新

### 4.2 类别ID对照表

| ID | 名称 | 说明 |
|----|------|------|
| 0 | 0惊喜/惊讶 | 强类, 用第二级 |
| 1 | 1恐惧 | 强类, 用第二级 |
| **2** | **2厌恶** | **弱类, 用第一级V17** |
| 3 | 3快乐/开心 | 强类, 用第二级 |
| **4** | **4悲伤** | **弱类, 用第一级V17** |
| 5 | 5愤怒 | 强类, 用第二级 |
| **6** | **6中性** | **弱类池, 但识别本身就强** |

### 4.3 V17内部标签映射

```
V17内部(0/1/2)  →  原始7类ID
    0           →   2 (厌恶)
    1           →   4 (悲伤)
    2           →   6 (中性)
```

### 4.4 推理速度参考（PC CPU）

| 操作 | 耗时 |
|------|------|
| flatten+scale+pca | <1ms |
| V17 SVM predict | ~0.3ms |
| V16 SVM predict | ~0.3ms |
| **单帧总计** | **<2ms** |

板子上可能稍慢，但远小于33ms/帧预算。

---

## 五、可选进阶优化

### 5.1 RF投票辅助（零成本提升悲伤~13pp）

如果同时加载了 `v16_rf_cleaned_56%`，可以在融合决策层加一条规则：

```python
pkg_rf = joblib.load('models/v16_rf_cleaned_56%_20260509_2136.joblib')
rf_model = pkg_rf['model']

# 在Step 3之后加:
rf_pred = rf_model.predict(x_50)[0]

# 规则: 如果RF说悲伤且V17不说悲伤 → 改判悲伤
if rf_pred == 4 and final_label != 4:
    final_label = 4  # RF对悲伤天然更敏感
```

预期效果：悲伤准确率再提升5-10pp。

### 5.2 调整置信度阈值

当前阈值 `CONF_THRESHOLD = 0.40`，可以调：
- **调高到0.55** → 更多样本降级为中性（保守）
- **调低到0.30** → 更少降级，更多保留V17判断（激进）
- 建议：先在验证集上试几个值看整体准确率变化

---

## 六、注意事项

1. **热力图尺寸必须是 (3, 128, 128)** — 这是训练时的输入规格
2. **不要重新fit PCA/Scaler** — 直接用joblib里的已拟合实例
3. **Python环境**: sklearn (推荐1.3.x+, 训练时用的1.3.2), numpy, joblib
4. **如果板子没有sklearn**: 需要安装 (`pip install scikit-learn`)
5. **V17不依赖PyTorch** — 纯sklearn模型，不需要GPU

---

## 七、问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `KeyError` 加载joblib | sklearn版本太旧 | 升级scikit-learn>=1.0 |
| 维度错误 | 热力图不是(3,128,128) | 检查shape |
| 结果全是同一类 | PCA/scaler没用对 | 确保用的是pkg里的transformer |
| 准确率很低 | 输入数据分布不同 | 检查热力图生成方式是否一致 |

---

> **策略完整文档**: 见 `docs/15_两级级联策略_厌恶悲伤中性.md`
> **PC端交接留痕**: 见项目根目录 `_HANDOVER_20260509.md`
