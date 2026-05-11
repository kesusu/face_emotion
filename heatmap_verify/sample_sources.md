# PC端验证样本清单 (128x128)

> 生成时间: 自动生成

## 用途

每个类别取1张原图 + 对应的128x128热力图缓存, 用于与板端对比验证MediaPipe一致性。

## 类别参数

- 热力图尺寸: 128x128
- 高斯sigma: 1.5
- 3通道: xy_position / z_depth / region_group
- 数据类型: float32
- 缓存格式: dict{'heatmap': (3,128,128), 'source': str}

## 样本列表

### 0惊讶

| 项目 | 值 |
|---|---|
| 原始图片 | `test_0004.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\0惊讶\test_0004.jpg` |
| 图片MD5 | `3a9e508b1c9d419874204d96e7941197` |
| 缓存文件 | `0惊讶_test_0004.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `12de7a35531f7ebd2ef95209094f5586` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\0惊讶\test_0004.jpg` |

### 1恐惧

| 项目 | 值 |
|---|---|
| 原始图片 | `test_0623.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\1恐惧\test_0623.jpg` |
| 图片MD5 | `303b733a05003978366af79238032db1` |
| 缓存文件 | `1恐惧_test_0623.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `3b7efca2cdc179d84a9478a429207bd8` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\1恐惧\test_0623.jpg` |

### 2厌恶

| 项目 | 值 |
|---|---|
| 原始图片 | `test_0007.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\2厌恶\test_0007.jpg` |
| 图片MD5 | `e3c6674a945b3f6a068be748705919a0` |
| 缓存文件 | `2厌恶_test_0007.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `41266fb151e770d73e336db75d61cc2c` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\2厌恶\test_0007.jpg` |

### 3快乐

| 项目 | 值 |
|---|---|
| 原始图片 | `test_0009.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\3快乐\test_0009.jpg` |
| 图片MD5 | `6b3ba25acad4d82abbef3df2975a382d` |
| 缓存文件 | `3快乐_test_0009.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `101ff7fa62d5c672eae38dcbcec39baa` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\3快乐\test_0009.jpg` |

### 4悲伤

| 项目 | 值 |
|---|---|
| 原始图片 | `test_0001.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\4悲伤\test_0001.jpg` |
| 图片MD5 | `0a252a4dd91527fc295ad409e011098d` |
| 缓存文件 | `4悲伤_test_0001.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `f2283207f58cc7f8fc25d4baf6c6e61e` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\4悲伤\test_0001.jpg` |

### 5愤怒

| 项目 | 值 |
|---|---|
| 原始图片 | `test_0017.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\5愤怒\test_0017.jpg` |
| 图片MD5 | `5e45628aa30fe019ffce33859c13c14f` |
| 缓存文件 | `5愤怒_test_0017.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `09c9efacc10cd80df121e370d66c4cd7` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\5愤怒\test_0017.jpg` |

### 6中性

| 项目 | 值 |
|---|---|
| 原始图片 | `test_2389.jpg` |
| 图片来源(PC) | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\6中性\test_2389.jpg` |
| 图片MD5 | `6fa84415cfd41993692a782924c5e89b` |
| 缓存文件 | `6中性_test_2389.jpg.npy` |
| 热力图shape | `(3, 128, 128)` |
| 热力图dtype | `float32` |
| npy MD5 | `fc3ec54bebea35573fc574adb677df70` |
| source标签 | `C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\6中性\test_2389.jpg` |

## 板端验证方法

```bash
# 1. 把整个 heatmap_verify/ 目录推送到板子
# 2. 在板子上对每张 original_xxx.jpg 用板端SDK做人脸关键点检测
# 3. 用相同参数(sigma=1.5, size=128)生成热力图
# 4. 对比 numpy 数值是否一致
```

