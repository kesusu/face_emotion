#!/usr/bin/env python3
"""
PC端批量热力图生成器 — 方案B (128x128)
=========================================
用MediaPipe替代板端fiboaisdk, 对RAF-DB/train全量生成热力图

用法:  py -3.11 batch_generate_heatmaps.py
输出: RAF-DB/train/reports/cnn_train_data_128.npy (~1-2GB)
"""

import os, sys, time, json, glob

# ====== 路径配置 ======
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RAF_DB_TRAIN = os.path.join(WORK_DIR, 'RAF-DB', 'train')
OUTPUT_DIR = os.path.join(RAF_DB_TRAIN, 'reports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 类别映射 (与cnn_heatmap_classifier.py一致)
CATEGORIES = {
    '0惊讶': '惊讶', '1恐惧': '恐惧', '2厌恶': '厌恶',
    '3快乐': '开心', '4悲伤': '悲伤', '5愤怒': '愤怒', '6中性': '中性'
}
LABEL_ORDER = ['惊讶', '恐惧', '厌恶', '开心', '悲伤', '愤怒', '中性']
LABEL2IDX = {e: i for i, e in enumerate(LABEL_ORDER)}

HEATMAP_SIZE = 128   # 方案B: 升级到128
GAUSS_SIGMA = 1.5    # 保持不变


# ==================== MediaPipe Landmarks提取 ====================
def create_landmarker():
    """创建FaceLandmarker实例 (复用)"""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    import mediapipe as mp

    model_path = os.path.expanduser('~/.mediapipe/models/face_landmarker.task')
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        print("请先运行: py -3.11 -c \"import urllib.request; ...\" 下载模型")
        sys.exit(1)

    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options), mp


def extract_landmarks(landmarker, mp_obj, image_path):
    """
    用MediaPipe提取468个关键点
    返回: (468, 3) 或 None
    """
    import cv2
    import numpy as np

    # 中文路径兼容
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp_obj.Image(image_format=mp_obj.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)

    if not results.face_landmarks or len(results.face_landmarks) == 0:
        return None

    landmark = results.face_landmarks[0]
    points = [[lm.x, lm.y, lm.z] for lm in landmark]
    arr = np.array(points, dtype=np.float32)

    # 新版MediaPipe输出478点, 截断到468
    if arr.shape[0] >= 468:
        arr = arr[:468]

    return arr


# ==================== 热力图生成 (从cnn_heatmap_classifier.py复制) ====================
def landmarks_to_heatmap(lm, size=HEATMAP_SIZE, sigma=GAUSS_SIGMA):
    """将468个3D关键点转为3通道热力图"""
    import numpy as np

    lm = np.asarray(lm, dtype=np.float64)
    if lm.shape[0] != 468:
        return np.zeros((3, size, size), dtype=np.float32)

    x = lm[:, 0] * (size - 1)
    y = lm[:, 1] * (size - 1)
    z = lm[:, 2]

    xx = np.arange(size, dtype=np.float64)
    yy = np.arange(size, dtype=np.float64)
    gx, gy = np.meshgrid(xx, yy)
    sigma_px = sigma * size / 10

    # Ch0: xy位置
    h0 = np.zeros((size, size), dtype=np.float64)
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            h0 += np.exp(-d2 / (2 * sigma_px**2))

    # Ch1: z深度
    h1 = np.zeros((size, size), dtype=np.float64)
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            weight = (z[i] + 1) / 2
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            h1 += weight * np.exp(-d2 / (2 * sigma_px**2))

    # Ch2: 区域分组
    h2 = np.zeros((size, size), dtype=np.float64)
    REGIONS = {
        'eyebrow': list(range(70,108)) + list(range(300,338)),
        'eyes': [33,133,159,145,386,374,263,362] + list(range(33,134)) + list(range(362,394)),
        'nose': [1,2,98,327,168,6,197,195,5],
        'mouth': list(range(61,92)) + list(range(291,322)) + [13,14,17,0,37,84,87,178,409,375,409,270],
        'contour': [10,109,67,103,54,21,162,127,237,137,452,356,345,
                    132,172,148,152,377,400,378,382,381,380,361],
    }
    region_weights = {'eyebrow': 2.0, 'eyes': 2.5, 'nose': 1.0, 'mouth': 3.0, 'contour': 0.8}
    for region_name, indices in REGIONS.items():
        w = region_weights.get(region_name, 1.0)
        for i in indices:
            if i >= 468: continue
            if 0 <= x[i] < size and 0 <= y[i] < size:
                d2 = (gx - x[i])**2 + (gy - y[i])**2
                h2 += w * np.exp(-d2 / (2 * sigma_px**2))

    def norm(h):
        mx = h.max()
        return (h / mx if mx > 0 else h).astype(np.float32)

    return np.stack([norm(h0), norm(h1), norm(h2)], axis=0)


# ==================== 主流程 ====================
def main():
    import numpy as np

    print("=" * 60)
    print("PC端批量热力图生成 — 方案B (128x128)")
    print("=" * 60)
    print(f"数据源: {RAF_DB_TRAIN}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"热力图尺寸: {HEATMAP_SIZE}x{HEATMAP_SIZE}")
    print(f"Sigma: {GAUSS_SIGMA}")

    # 统计图片总数
    total_images = 0
    class_info = {}
    for item in sorted(os.listdir(RAF_DB_TRAIN)):
        dp = os.path.join(RAF_DB_TRAIN, item)
        if not os.path.isdir(dp): continue
        # 目录名直接匹配CATEGORIES key (如 "0惊讶", "1恐惧" ...)
        if item not in CATEGORIES: continue
        jpgs = sorted([f for f in os.listdir(dp) if f.endswith('.jpg')])
        class_info[item] = jpgs
        total_images += len(jpgs)
        print(f"  {item}: {len(jpgs)} 张")

    print(f"\n总计: {total_images} 张图片")
    print("-" * 60)

    # 初始化MediaPipe (只初始化一次!)
    print("\n[1/3] 初始化MediaPipe...")
    t_init = time.time()
    landmarker, mp_obj = create_landmarker()
    print(f"  初始化完成 ({time.time()-t_init:.1f}s)")

    # 增量检查: 是否已有缓存?
    cache_dir = os.path.join(OUTPUT_DIR, 'heatmap_cache_128')
    os.makedirs(cache_dir, exist_ok=True)

    save_path = os.path.join(OUTPUT_DIR, 'cnn_train_data_128.npy')
    
    # 如果已有完整npy, 检查是否需要重新生成
    if os.path.exists(save_path):
        existing = np.load(save_path, allow_pickle=True).item()
        existing_count = existing.get('total_samples', 0)
        if existing_count >= total_images:
            print(f"[INFO] 已有缓存: {save_path} ({existing_count}张)")
            print("[INFO] 强制重新生成 (方案B: 128x128全量)")
            # 不return, 继续生成

    # 逐类处理
    all_hmaps = []
    all_labels = []
    all_paths = []
    fail_count = 0
    cache_hit = 0
    cache_new = 0

    t_start = time.time()

    for cls_name, jpgs in class_info.items():
        true_label = CATEGORIES[cls_name]
        label_idx = LABEL2IDX[true_label]

        print(f"\n[处理] {cls_name} ({len(jpgs)}张)...", end=" ", flush=True)
        tc = time.time()
        cc_ok = 0
        cc_fail = 0

        for jname in jpgs:
            jpath = os.path.join(RAF_DB_TRAIN, cls_name, jname)

            # 检查单张缓存
            cache_file = os.path.join(cache_dir, f"{cls_name}_{jname}.npy")
            if os.path.exists(cache_file):
                try:
                    cached = np.load(cache_file, allow_pickle=True)
                    if isinstance(cached, dict) and 'heatmap' in cached:
                        all_hmaps.append(cached['heatmap'])
                        all_labels.append(label_idx)
                        all_paths.append(jpath)
                        cache_hit += 1
                        cc_ok += 1
                        continue
                except:
                    pass

            # 提取landmarks
            lm = extract_landmarks(landmarker, mp_obj, jpath)
            if lm is None:
                cc_fail += 1
                fail_count += 1
                continue

            # 生成热力图
            hmap = landmarks_to_heatmap(lm)

            # 保存单张缓存
            try:
                np.save(cache_file, {'heatmap': hmap, 'source': jpath})
            except:
                pass  # 缓存保存失败不影响主流程

            all_hmaps.append(hmap)
            all_labels.append(label_idx)
            all_paths.append(jpath)
            cache_new += 1
            cc_ok += 1

        elapsed = time.time() - tc
        rate = len(jpgs) / elapsed if elapsed > 0 else 0
        print(f"OK:{cc_ok} FAIL:{cc_fail} ({elapsed:.1f}s, {rate:.1f}张/s)")

    total_time = time.time() - t_start
    rate_overall = (total_images - fail_count) / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print(f"完成! 总计处理: {total_images-fail_count}/{total_images} 张")
    print(f"  成功: {total_images-fail_count}")
    print(f"  失败: {fail_count} (人脸检测失败)")
    print(f"  缓存命中: {cache_hit}, 新增: {cache_new}")
    print(f"  耗时: {total_time:.1f}s ({rate_overall:.1f} 张/s)")

    # 组装并保存最终数据
    print(f"\n[3/3] 保存训练数据...")
    heatmaps = np.stack(all_hmaps) if all_hmaps else np.zeros((0, 3, HEATMAP_SIZE, HEATMAP_SIZE))
    labels = np.array(all_labels, dtype=np.int64)

    result = {
        'heatmaps': heatmaps,
        'labels': labels,
        'features': None,
        'file_paths': all_paths,
        'label_order': LABEL_ORDER,
        'categories': CATEGORIES,
        'total_samples': len(labels),
        'heatmap_size': HEATMAP_SIZE,
        'sigma': GAUSS_SIGMA,
        'source': 'PC_MediaPipe_FaceLandmarker',
    }

    np.save(save_path, result)
    size_mb = os.path.getsize(save_path) / (1024*1024)
    print(f"\n已保存: {save_path} ({size_mb:.1f}MB)")
    print(f"  shape: {heatmaps.shape}")
    print(f"  类别分布:")

    class_counts = np.bincount(labels, minlength=7)
    for i in range(7):
        if class_counts[i] > 0:
            pct = class_counts[i]/len(labels)*100
            print(f"    {LABEL_ORDER[i]}: {class_counts[i]} 张 ({pct:.1f}%)")

    landmarker.close()
    print("\nDone! 下一步运行CNN训练:")
    print(f"  python cnn_heatmap_classifier.py --model-type deep --epochs 120 --batch-size 64 --lr 0.002")


if __name__ == '__main__':
    main()
