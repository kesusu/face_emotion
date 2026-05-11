# Face Emotion 项目进度记录
> 更新时间: 2026-05-09 01:09 (今日收工, 热力图+验证样本全部完成)

---

## 断点位置 📍
**✅ 热力图 + 验证样本全部完成！下一步：CNN训练**
- 已完成: 0惊讶 ~ 6中性 (**7/7 类**)
- 总计: **8603张成功**, 1303 FAIL (无人脸)
- 训练数据已保存: `RAF-DB/train/reports/cnn_train_data_128.npy` (1613.8MB)
- 验证样本已就绪: `heatmap_verify/` (**2MB**, 7类别×原图+热力图, 含MD5校验)

---

## 已完成 ✅

### 1. 数据污染排查与修复
- **问题**: PC端 `RAF-DB/train/0惊讶` 中混入了 **1672张** 类别4(悲伤)的错放文件
- **修复操作**:
  - 删除了 PC端 `RAF-DB/train/0惊讶` 中与4悲伤同名的 **1672个** 错放文件
  - 删除了 PC端 `train/0` 中与4悲伤同名的 **1631个** 错放文件
  - 以板子 `train2/0惊讶` 为准，删除PC端多出的 **31张** 用户判定为坏数据的图片
- **结果**: PC端 `RAF-DB/train/0惊讶` = **1575张**，与板子 train2 一致 ✅

### 2. 热力图缓存清理
- 清理了 **1498个** 失效热力图缓存（对应已删除的错放图片）

### 3. 热力图增量生成 ✅ 全部完成
- 0惊讶: OK:1430 FAIL:145
- 1恐惧: OK:288 FAIL:44
- 2厌恶: OK:754 FAIL:108
- 3快乐: OK:980 FAIL:128
- 4悲伤: OK:1893 FAIL:248
- 5愤怒: OK:623 FAIL:191
- 6中性: OK:2635 FAIL:439
- **总计: 8603 OK / 1303 FAIL, 耗时1440.5s**
- **训练数据已保存: cnn_train_data_128.npy (1613.8MB, shape=(8603,3,128,128))**

### 4. 文档整理
- 所有MD归入 `docs/` 文件夹
- 根目录新建 `README.md` 项目说明
- 新增 `docs/10_数据污染修复报告_给板子.md`
- 新增 `docs/11_板端推送清单.md`
- 整理了 `_noface_0surprise.txt`（145张无人脸图片列表）

### 5. 验证样本提取 ✅ 全部完成
- 从热力图缓存中提取 **7个类别各1个样本** 到 `heatmap_verify/`
- 每个类别包含:
  - `original_xxx.jpg` — 原始图片（标注了PC端来源路径 + MD5）
  - `heatmap_cache_128.npy` — 128x128热力图 (3,128,128) float32, sigma=1.5
- 已清理旧版64x64残留文件，全部统一为 **128x128格式**
- 完整来源清单+MD5校验: `heatmap_verify/sample_sources.md`
- 总大小仅 **2MB**，方便推送到板子做MediaPipe一致性对比

| 类别 | 原图 | 来源路径 |
|------|------|----------|
| 0惊讶 | test_0004.jpg | RAF-DB/train/0惊讶/test_0004.jpg |
| 1恐惧 | test_0623.jpg | RAF-DB/train/1恐惧/test_0623.jpg |
| 2厌恶 | test_0007.jpg | RAF-DB/train/2厌恶/test_0007.jpg |
| 3快乐 | test_0009.jpg | RAF-DB/train/3快乐/test_0009.jpg |
| 4悲伤 | test_0001.jpg | RAF-DB/train/4悲伤/test_0001.jpg |
| 5愤怒 | test_0017.jpg | RAF-DB/train/5愤怒/test_0017.jpg |
| 6中性 | test_2389.jpg | RAF-DB/train/6中性/test_2389.jpg |

---

## 下次继续 📋

### [x] ✅ 完成热力图生成（7/7类全部完成）
### [x] ✅ 提取验证样本到 heatmap_verify/（7类别 × 2文件 = 14个, 2MB）
### [x] ✅ 用干净数据重新训练 CNN 模型 (V11最佳: 52.1% ValAcc)
### [x] ✅ FAIL数据分析 + 记录 (1303张, 13.1%, 见 docs/12_FAIL数据分析报告.md)
### [x] ✅ V11 CNN训练经验完整总结 (见 docs/13_CNN训练经验完整总结_V11.md)
> ⚠️ V11模型因PC/板子MediaPipe版本差异不可用
> 📋 待板子推送新兼容热力图后 → 按13号文档直接复现训练

**V11配置**: AdamW + OneCycleLR, 200ep, lr=1e-3, dropout=0.5, label_smoothing=0.05
**模型已推送**: `from_pc/v11_cnn_deep_best_128.pth` (5.8MB)
**验证样本已推送**: `from_pc/heatmap_verify_128/` (17文件, 1.4MB)

> ⚠️ 待确认：板子端MediaPipe可检测出PC端FAIL的1303张 → 存在环境差异
> 📋 后续可选：在板子补回这1303张热力图后重新训练（数据量+14%）

### [x] ✅ 推送到板子（2样东西已全部推送）
1. ✅ **v11_cnn_deep_best_128.pth** (5.8MB, 52.1% ValAcc) → `from_pc/`
2. ✅ **heatmap_verify_128/** (1.4MB, 7类别验证样本) → `from_pc/`

> 不需要推送：0惊讶图片（train2已有）、cnn_train_data_128.npy（板子用jpg推理）、全量缓存

### [ ] 更新文档最终状态

---

## 关键路径

```
c:\Users\ke\Desktop\嵌赛\face_emotion\
├── docs/                           # 所有文档 + todo.md
│   ├── 10_数据污染修复报告_给板子.md   # ⭐ 重要：告诉板子发生了什么
│   └── 11_板端推送清单.md             # ⭐ 推送什么、怎么推
├── README.md                       # 项目说明
├── RAF-DB\train\                   # 干净的训练数据
│   └── reports\
│       ├── heatmap_cache_128\      # 热力图缓存（7/7类全部完成 ✅）
│       └── cnn_train_data_128.npy  # ✅ 已保存: 8603样本, 1613.8MB
├── heatmap_verify/                 # ✅ PC vs 板端 验证样本 (2MB)
│   ├── sample_sources.md           # 来源清单+MD5校验
│   ├── 0惊讶/ ~ 6中性/             # 每类: 原图 + 128x128热力图npy
├── _noface_0surprise.txt           # 0惊讶中145张无人脸图片列表
```

---

## 注意事项
- ⚠️ 之前训练的 v9_cNN_deep_best.pth 是基于**污染数据**的，不可信
- ⚠️ 板子 photos/train/0惊讶 含1683个错放文件，不要用；用 train2/
- ✅ 热力图参数: 128x128, sigma=1.5, float32, 3通道
- ✅ MediaPipe 报错 portable_clearcut... 是网络遥测问题，**无影响**
