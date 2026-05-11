# V16+V17 部署包 README (板子SDK数据重训版)

> 时间: 2026-05-09 22:22
> 数据源: **board_sdk_fiboaisdk** (cnn_train_data_128.npy, 9906张)
> 已删除: heatmap_cache_128/ (PC MediaPipe热力图, 禁止使用!)

---

## 文件清单

```
from_pc/
├── README_V17_DEPLOY_BOARD_SDK.md    ← 本文件
│
├── models/
│   ├── v16_svm_boardSDK_42%_20260509_2222.joblib   (~11.5MB)  ★ 7类SVM基线
│   ├── v16_rf_boardSDK_47%_20260509_2222.joblib     (~30.7MB)  ★ 7类RF(中性87%)
│   └── v17_3class_svm_boardSDK_55%_20260509_2222.joblib (~10.8MB) ★★★ 三类专用
│
└── docs/
    └── 15_两级级联策略.md              完整策略设计文档
```

---

## 关键信息: 标签编码 (0-based!)

```
0 = 惊讶 (1575张)
1 = 恐惧 ( 332张)
2 = 厌恶 ( 862张)
3 = 开心 (1108张)
4 = 悲伤 (2141张)
5 = 愤怒 ( 814张)
6 = 中性 (3074张)
合计: 9906张
```

---

## 模型效果总结

### V16a: SVM(RBF,C=5) - 7类全量
- 总准确率: **41.7%** (验证集1982张)
- 强项: 惊讶49.5%, 开心50%, 愤怒52.1%
- 弱项: 悲伤18.2%, 恐惧33.3%

### V16b: RF(200树) - 7类
- 总准确率: **47.2%**
- 超强项: **中性 87.0%** (可用来修正中性判断)
- 弱项: 恐惧1.5%, 厌恶3.5%

### V17: 三类专用SVM (厌恶/悲伤/中性) ⭐ 核心模型
- 总准确率: **54.5%** (仅3类)
- 厌恶: **57.8%** (vs V16的41%, +17pp)
- 悲伤: **43.2%** (vs V16的18%, +25pp)
- 中性: **61.5%** (vs V16的49%, +12pp)

---

## 两级级联策略 (核心思路)

```
板子fiboaisdk生成热力图(3,128,128)
        ↓ 展平 → (49152,)
        ↓ scaler.transform → pca.transform → (50,)
        ↓
   ┌─────────────────────────────────────┐
   │ [第二级] V16-SVM 7类预测             │
   │ 结果 ∈ {惊讶,恐惧,开心,愤怒}         │
   │      → 直接输出 (板子原本就强的!)    │
   │                                     │
   │ 结果 ∈ {厌恶,悲伤,中性}              │
   │      → 送入第一级V17重判            │
   └──────────┬──────────────────────────┘
              ↓
   ┌─────────────────────────────────────┐
   │ [第一级] V17 三类专用SVM            │
   │ 输入: 同样的PCA50特征               │
   │ 输出: 厌恶 / 悲伤 / 中性            │
   │ 效果: 厌恶58% / 悲伤43% / 中性62%   │
   └─────────────────────────────────────┘
              ↓
        最终预测结果
```

**预期融合后全体准确率**: ~50-55% (厌恶和悲伤大幅改善)

---

## 板端推理代码模板

```python
import numpy as np
import joblib

# ========== 加载模型 ==========
# 注意: V16和V17共用同一个PCA+Scaler!
pkg_v16 = joblib.load('models/v16_svm_boardSDK_42%_20260509_2222.joblib')
pkg_v17 = joblib.load('models/v17_3class_svm_boardSDK_55%_20260509_2222.joblib')

svm_7class = pkg_v16['model']       # SVC 7分类器
svm_3class = pkg_v17['model']       # SVC 3分类器(厌恶/悲伤/neutral)
pca = pkg_v16['pca']                # PCA50 (V16/V17共用!)
scaler = pkg_v16['scaler']          # StandardScaler (共用!)

label_names_7 = pkg_v16['label_names']  # {0:'惊讶', 1:'恐惧', ..., 6:'中性'}
lmap_rev = pkg_v17['label_map_rev']     # {0:2, 1:4, 2:6}
names_3 = pkg_v17['class_names_3']      # {2:'厌恶', 4:'悲伤', 6:'中性'}

# ========== 推理函数 ==========
def predict_cascade(heatmap_3x128x128):
    """
    输入: heatmap (3, 128, 128) - 从板子fiboaisdk获取
    输出: (预测标签ID, 预测名称)
    """
    # 特征提取 (与训练时完全一致!)
    x = heatmap_3x128x128.reshape(1, -1).astype(np.float32)  # (1, 49152)
    x = scaler.transform(x)      # StandardScaler
    x_pca = pca.transform(x)     # PCA50 -> (1, 50)

    # 第二级: 7类SVM
    pred_7 = svm_7class.predict(x_pca)[0]
    pred_name_7 = label_names_7[pred_7]

    # 融合决策
    if pred_7 in {0, 1, 3, 5}:  # 惊讶/恐惧/开心/愤怒 → 板子强的直接用
        return int(pred_7), pred_name_7

    # 第一级: V17三类专用 (厌恶/悲伤/中性)
    pred_3_internal = svm_3class.predict(x_pca)[0]  # 0/1/2
    pred_final = lmap_rev[pred_3_internal]           # 2/4/6
    return int(pred_final), names_3[pred_final]

# ========== 使用示例 ==========
# heatmap 来自 fiboaisdk 的输出, shape=(3, 128, 128)
# label_id, label_name = predict_cascade(heatmap)
# print(f"预测: {label_name}({label_id})")
```

---

## 性能参考

| 操作 | 耗时 |
|------|------|
| Scaler+PCA变换 | <0.5ms |
| SVM predict (7类) | ~1ms |
| SVM predict (3类) | ~0.3ms |
| **单帧总推理** | **<2ms** (纯CPU, 不需要GPU/TensorFlow) |

## 注意事项

1. **热力图来源必须一致**: 训练和推理都用 board_sdk_fiboaisdk 生成的热力图
2. **不要用PC MediaPipe生成的热力图**: 已全部删除heatmap_cache_128/
3. **V16和V17共用同一个PCA和Scaler**: 加载时注意从V16包取pca/scaler，或V17包里的也一样(复制的)
4. **标签是0-based**: 0=惊讶 ... 6=中性
