# face_emotion 项目目录 & 文件说明

> **最后更新**: 2026-05-09 21:32 (数据清洗后RF重训进行中)
> **配合阅读**: `docs/00_文档索引.md` (文档导航) + `README.md` (项目简介) + `_HANDOVER_20260509.md` (最新交接留痕 ★)

---

## 一、顶层目录结构总览

```
face_emotion/                          ← 本项目根目录
│
├── 📁 docs/                           ★ 所有文档 (14个md)
├── 📁 RAF-DB/train/                   ⭐ 干净训练数据 (9906张jpg + 8603热力图npy)
├── 📁 train/                          数据副本 (11672张jpg, 未完全同步RAF-DB)
├── 📁 photos/                         板子原始数据备份 (含污染)
│   └── train/reports/                 ⚠️ 旧训练产出(已归档到 _废弃_/)
│
├── 📁 heatmap_verify/                 PC vs 板端 验证样本 (7类别×2文件, ~2MB)
├── 📁 ___AI_CNN训练交接包___/          ADB拉取的板子原始交接包
├── 📁 _废弃_污染数据与旧模型_待手动删除/ 🗑️ 污染数据+过时模型(确认后可删)
│
├── 🐍 cnn_heatmap_classifier.py      CNN训练主脚本 (DeepCNN, V11配置)
├── 🐍 batch_generate_heatmaps.py     热力图批量生成器 (MediaPipe→128x128高斯)
├── 🐍 heatmap_verify.py              热力图验证工具脚本
├── 🐍 extract_features.py            特征提取工具
├── 🐍 sync_data.py                   ADB同步数据脚本
├── 🐍 _check_mix.py                  数据交叉检查脚本
│
├── 📄 v11_svm_improved.pkl           ✅ SVM模型 (balanced, 47.3% ValAcc, 660KB)
├── 📄 v11_svm_report.json            ✅ SVM训练报告
├── 📄 v9_cnn_deep_best_v95_sgd.pth   ⚠️ 旧V9模型(39.7%, 已归档到 _废弃_/)
│
├── 📄 test_0010.jpg                  测试图片
├── 📄 pc_verify_package.zip          PC验证包
│
├── 📋 cnn_train_log.txt              CNN训练日志
├── 📋 heatmap_gen_log.txt            热力图生成日志
├── 📋 mediapipe-face关键点详解与情绪识别参数.txt  MediaPipe技术参考
│
├── 📋 _board_0.txt                   板子0惊讶文件列表
├── 📋 _pc0_list.txt                  PC端0惊讶文件列表
├── 📋 _check_0.txt                   对比检查结果
├── 📋 _noface_0surprise.txt          0惊讶中145张无人脸图片列表
│
├── 📘 README.md                      ★ 项目简介 (必读)
├── 📘 本文件                         ★ 目录&文件说明 (本文件)
├── 📘 _HANDOVER_20260509.md          ★ 最新交接留痕 (★ 下一个AI先读这个!)
│
├── 🐍 _rf_cleaned_retrain.py         RF清洗后重训脚本 (正在运行/刚跑)
├── 🐍 _run_rf.py                     RF启动包装器 (重定向输出)
```

---

## 二、核心数据目录详情

### `RAF-DB/train/` — ⭐ 主训练数据源（用户2026-05-09手工清洗后 ✅）

```  
RAF-DB/train/
├── 0惊讶/    ~1575 张 jpg   → 1430 个热力图
├── 1恐惧/     ~332 张 jpg   →  288 个热力图
├── 2厌恶/     ~862 张 jpg   →  754 个热力图
├── 3快乐/    ~1108 张 jpg   →  980 个热力图
├── 4悲伤/    ~1698 张 jpg   → 1514 个热力图 (用户删~240张)
├── 5愤怒/     ~814 张 jpg   →  623 个热力图
├── 6中性/    ~???  张 jpg   → ~2486个热力图 (用户删~6张, 待确认)
│
└── reports/
    ├── heatmap_cache_128/             # ~8603+ 个 .npy (扁平结构! ⚠️)
    │   # ⚠️ npy在根目录下, 不是子目录!
    │   # 命名格式: {类别}_{jpg文件名}.npy
    │   # 例: 4悲伤_train_0442.jpg.npy
    │
    ├── cnn_train_data_128.npy         # ⚠️ 旧数据(9906张未清洗, 1858MB), 不要用!
    │                                   # 应基于JPG现扫+匹配npy
    ├── v15_boardSDK_deep_acc306_ep171.pth  # V15 CNN最佳 (5.7MB)
    ├── v15_cnn_deep_report_boardSDK.json   # V15报告 (16.8KB)
    ├── v9_cnn_deep_best.pth            # V11 CNN (5.8MB, 命名遗留=实际V11)
    └── v9_cnn_report.json             # V11训练报告
```

### `train/` — 数据副本

| 说明 | 详情 |
|------|------|
| 来源 | 早期从板子ADB pull的副本 |
| 图片数 | 11672张 |
| 状态 | ⚠️ 未与RAF-DB完全同步，部分含污染 |
| 用途 | 开发调试用，**不用于正式训练** |

### `photos/` — 板子原始数据备份

```
photos/
└── train/
    ├── 0惊讶/  ...                    # 含1683个错放文件! ⚠️ 污染!
    ├── reports/
    │   ├── cnn_train_data.npy        # 215MB, 64x64, 污染数据 (已归档)
    │   ├── v9_cnn_deep_best.pth       # 3.7MB, 污染训练 (已归档)
    │   ├── v9_cnn_deep_best_board.pth # 5.6MB, 同上 (已归档)
    │   ├── v9_cnn_light_best.pth      # 1.8MB, 同上 (已归档)
    │   ├── v9_cnn_report.json        # 旧报告 (已归档)
    │   ├── heatmap_cache/            # 旧64x64缓存 (空壳, 已归档)
    │   └── heatmap_cac/              # 旧缓存别名 (已归档)
```

> 整个 `photos/train/reports/` 内容已复制到 `_废弃_污染数据与旧模型_待手动删除/photos_train_reports_旧/`

---

## 三、验证样本目录

### `heatmap_verify/` — PC vs 板端一致性验证

```
heatmap_verify/
├── sample_sources.md                  # 来源清单 + MD5校验值
├── verify_report.txt
├── verify_results.json
│
├── 0惊讶/
│   ├── original_test_0004.jpg         # 原始图片 (来源: RAF-DB/train/0惊讶/)
│   └── heatmap_cache_128.npy          # 128x128热力图 shape=(3,128,128) float32
├── 1恐惧/  → original_test_0623.jpg + heatmap_cache_128.npy
├── 2厌恶/  → original_test_0007.jpg + heatmap_cache_128.npy
├── 3快乐/  → original_test_0009.jpg + heatmap_cache_128.npy
├── 4悲伤/  → original_test_0001.jpg + heatmap_cache_128.npy
├── 5愤怒/  → original_test_0017.jpg + heatmap_cache_128.npy
└── 6中性/  → original_test_2389.jpg + heatmap_cache_128.npy
```

---

## 四、Python 脚本说明

| 文件 | 功能 | 运行环境 |
|------|------|---------|
| **`cnn_heatmap_classifier.py`** | ⭐ CNN训练主脚本 (DeepCNN架构, 支持deep/light两种) | `py -3.8` (需sklearn) |
| **`batch_generate_heatmaps.py`** | ⭐ MediaPipe人脸检测→468关键点→高斯热力图(128x128), 支持增量跳过缓存 | `py -3.11` |
| `heatmap_verify.py` | 从已有缓存提取验证样本, 生成MD5校验 | `py -3.x` |
| `extract_features.py` | 特征提取 (SVM用14维特征或CNN用热力图) | `py -3.x` |
| `sync_data.py` | ADB数据同步辅助脚本 | `py -3.x` |
| `_check_mix.py` | 数据交叉检查 (排查0惊讶污染时用) | `py -3.x` |

---

## 五、关键产出物速查（当前有效 ✅ 2026-05-09 21:36更新）

| 物品 | 路径 | 大小 | 说明 |
|------|------|------|------|
| **★ 最佳模型: SVM(RBF,C=5)+PCA50** | `RAF-DB/train/reports/v16_svm_cleaned_56%_20260509_2136.joblib` | **13.4 MB** | **56.0% ValAcc ★ 新纪录**, 含pca+scaler+标签映射 |
| **RF(200树)+PCA50 (第二)** | `RAF-DB/train/reports/v16_rf_cleaned_56%_20260509_2136.joblib` | **83.1 MB** | **55.5% ValAcc**, 含pca+scaler |
| **KNN(k=21,cos)+PCA50** | `RAF-DB/train/reports/v16_knn_cleaned_48%_20260509_2136.joblib` | **12.3 MB** | 49.4% ValAcc, 含pca+scaler |
| V15 CNN模型(板子SDK) | `RAF-DB/train/reports/v15_boardSDK_deep_acc306_ep171.pth` | 5.7 MB | 30.6% ValAcc, 已推送板子(已过时) |
| CNN最佳(V11) | `RAF-DB/train/reports/v9_cnn_deep_best.pth` | 5.8 MB | 52.1% ValAcc(可能过拟合) |
| V16训练报告 | `RAF-DB/train/reports/rf_cleaned_v2_report.json` | ~3 KB | 完整对比+逐类准确率 |
| V15 CNN训练报告 | `RAF-DB/train/reports/v15_cnn_deep_report_boardSDK.json` | 16.8 KB | CNN逐类准确率 |
| 训练数据集(JPG) | `RAF-DB/train/{0~6类别}/` | ~9244张jpg | 用户手工清洗后(2026-05-09) |
| 热力图缓存 | `RAF-DB/train/reports/heatmap_cache_128/` | ~1.6 GB | ~8050个npy(dict格式), 1169缺失 |
| SVM旧模型(手工特征) | `v11_svm_improved.pkl` | 660 KB | 54.5%(旧数据, 未清洗, 已被超越) |
| 板端部署包(CNN) | `_board_deploy_package/` | ~11.5 MB | V15 CNN + 推理脚本(PC端副本) |

---

## 六、废弃/归档文件 (`_废弃_/`)

详见该文件夹内 `README_清单说明.md`

### 🗑️ 污染数据产物（可删）

| 归档内容 | 大小 | 原因 |
|----------|------|------|
| `photos_train_reports_旧/` 全部内容 | ~230 MB | 基于数据污染(0惊讶混入1672张错放), 含cnn_train_data.npy/旧模型/旧报告 |

### ⏪ 过时但可用（可留）

| 文件 | 大小 | 说明 |
|------|------|------|
| `v9_cnn_deep_best_v95_sgd.pth`(根目录) | 3.7 MB | 干净数据训练, V9=39.7%, 被V11=52.1%取代, 可作回退方案 |

> **操作**: 整文件夹删掉或只删污染部分，见 `README_清单说明.md` 的删除建议

---

## 七、文档目录 (`docs/`) 完整列表

| 序号 | 文件名 | 核心内容 | 必读对象 |
|------|--------|---------|---------|
| 00 | `00_文档索引.md` | ★ 全部文档导航入口 | 所有人 |
| 01 | `01_AI项目交接文档_给开发板.md` | 项目初始交接 | 板端AI |
| 02 | `02_AI优化策略与提示词.md` | 混淆情绪突破策略 | PC+板端 |
| 03 | `03_AI接手交接说明_给电脑.md` | PC接手说明 | PC端AI |
| 04 | `04_AI_CNN训练报告_PC端.md` | CNN方案对比 | PC+板端 |
| 05 | `05_板端接手交接文档.md` | 板端环境全貌 | **板端AI必读** |
| 06 | `06_全局开发总结与突破方向.md` | 三引擎全局分析 | PC+板端 |
| 07 | `07_PC替代板端热力图验证方案.md` | 热力图验证方案 | PC端AI |
| 08 | `08_板端协同分工任务单.md` | ★ Ensemble执行步骤 | **板端AI必读** |
| 09 | `09_PC端总结与路线图.md` | PC端工作总结 | PC+板端 |
| 10 | `10_数据污染修复报告_给板子.md` | 污染修复过程 | 板端AI |
| 11 | `11_板端推送清单.md` | 推送操作步骤 | PC+板端 |
| 12 | `12_FAIL数据分析报告.md` | 🆕 1303张FAIL分析+补回方案 | PC+板端 |
| — | `todo.md` | 当前进度追踪 | PC+板端 |

---

## 八、板子推送记录与路径规范（★ 必读，禁止搞错）

### ⚠️ ADB推送路径规范（2026-05-09 确认，违反必错）

| 项目 | 值 |
|------|-----|
| ADB设备号 | `28de40d2` (qcs6490, 通过USB连接PC) |
| 板子工作目录 | `/home/fibo/AI model/cv_models/pose/` |
| **PC→板子 推送目标目录** | **`/home/fibo/AI model/cv_models/pose/___AI_CNN训练交接包___/from_pc/`** |
| ❌ 绝对禁止推送到 | `/data/local/tmp/pose/` (这是tmp临时目录，不是工作目录!) |

**标准推送命令模板：**
```powershell
# 推送模型
adb -s 28de40d2 push "<本地文件>" "/home/fibo/AI model/cv_models/pose/___AI_CNN训练交接包___/from_pc/models/"

# 推送脚本
adb -s 28de40d2 push "<本地文件>" "/home/fibo/AI model/cv_models/pose/___AI_CNN训练交接包___/from_pc/scripts/"

# 推送报告
adb -s 28de40d2 push "<本地文件>" "/home/fibo/AI model/cv_models/pose/___AI_CNN训练交接包___/from_pc/reports/"

# 推送文档(放from_pc根目录)
adb -s 28de40d2 push "<本地文件>" "/home/fibo/AI model/cv_models/pose/___AI_CNN训练交接包___/from_pc/"

# 验证推送结果
adb -s 28de40d2 shell "ls -laR '/home/fibo/AI model/cv_models/pose/___AI_CNN训练交接包___/from_pc/'"
```

### 历史推送记录

| 批次 | 时间 | 内容 | 目标路径 |
|------|------|------|---------|
| push_01 | 05-08~09 | V11模型 + heatmap_verify_128 + 早期散放文件 | `from_pc/` 根目录 |
| push_02 | 05-09 09:55 | 6个md文档(todo/FAIL报告/优化策略等) | `from_pc/push_02_FAIL分析与文档更新_20260509/docs/` |
| push_03 | 05-09 20:58 | ★ V15 CNN模型(30.6%) + 推理脚本 + 训练报告 + README | `from_pc/{models,scripts,reports}/` |

---

## 九、临时/调试文件

| 文件 | 用途 | 可删? |
|------|------|-------|
| `_board_0.txt` | 排查污染时板子0惊讶文件列表 | 可删 |
| `_pc0_list.txt` | 排查污染时PC端0惊讶文件列表 | 可删 |
| `_check_0.txt` | 交叉比对结果 | 可删 |
| `_noface_0surprise.txt` | 0惊讶145张无人脸图片列表 | 保留备用 |
| `test_0010.jpg` | 测试用单张图片 | 可删 |
| `cnn_train_log.txt` | 训练终端输出日志 | 可删 |
| `heatmap_gen_log.txt` | 热力图生成终端日志 | 可删 |
| `pc_verify_package.zip` | PC验证压缩包 | 可删 |
| `mediapipe-face关键点详解与情绪识别参数.txt` | 技术参考 | 保留 |
| `___AI_CNN训练交接包___/` | 从板子拉取的原始交接包 | 可删 |

---

*本文件由PC端AI生成 | 与 README.md 配合使用: README=项目概览, 本文件=目录细节*
