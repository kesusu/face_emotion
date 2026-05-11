#!/usr/bin/env python3
"""
实验1: PC端MediaPipe批量生成高质量热力图 (适配train/数字目录)
=============================================================
输出: RAF-DB/train/reports/cnn_train_data_pc_mediapipe.npy
"""
import os, sys, time, glob, json, math

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(WORK_DIR, 'train')  # train/0 ~ train/6
OUTPUT_DIR = os.path.join(WORK_DIR, 'RAF-DB', 'train', 'reports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数字目录 → 中文标签映射
NUM_TO_CN = {
    0: ('0惊讶', '惊讶'), 1: ('1恐惧', '恐惧'),
    2: ('2厌恶', '厌恶'), 3: ('3快乐', '开心'),
    4: ('4悲伤', '悲伤'), 5: ('5愤怒', '愤怒'),
    6: ('6中性', '中性'),
}
LABEL_ORDER = ['惊讶','恐惧','厌恶','开心','悲伤','愤怒','中性']
LABEL2IDX = {e:i for i,e in enumerate(LABEL_ORDER)}

HEATMAP_SIZE = 128
GAUSS_SIGMA = 1.5

# ==================== MediaPipe Landmarker ====================
def create_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    model_path = os.path.expanduser('~/.mediapipe/models/face_landmarker.task')
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    return landmarker, mp


def extract_landmarks(landmarker, mp_obj, image_path):
    """提取468个3D关键点"""
    import cv2, numpy as np
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
    if arr.shape[0] >= 468:
        arr = arr[:468]
    return arr


# ==================== 热力图生成 ====================
def landmarks_to_heatmap(lm, size=HEATMAP_SIZE, sigma=GAUSS_SIGMA):
    import numpy as np
    lm = np.asarray(lm, dtype=np.float64)
    if lm.shape[0] != 468:
        return np.zeros((3, size, size), dtype=np.float32)
    x = lm[:, 0] * (size - 1)
    y = lm[:, 1] * (size - 1)
    z = lm[:, 2]
    z = lm[:, 2]
    xx = np.arange(size, dtype=np.float64)
    yy = np.arange(size, dtype=np.float64)
    gx, gy = np.meshgrid(xx, yy)
    sigma_px = sigma * size / 10
    
    h0 = np.zeros((size, size), dtype=np.float64)
    h1 = np.zeros((size, size), dtype=np.float64)
    h2 = np.zeros((size, size), dtype=np.float64)
    
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            g = np.exp(-d2 / (2 * sigma_px**2))
            h0 += g
            w = (z[i] + 1) / 2
            h1 += w * g
    
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


def main():
    import numpy as np
    
    print("="*60)
    print("实验1: PC MediaPipe批量生成高质量热力图")
    print("="*60)
    print(f"数据源: {TRAIN_DIR}/")
    print(f"输出: {OUTPUT_DIR}/cnn_train_data_pc_mediapipe.npy")
    
    landmarker, mp_obj = create_landmarker()
    print("[MediaPipe] 初始化完成")
    
    all_hmaps = []
    all_labels = []
    total = 0
    fail = 0
    per_class = {}
    t0_total = time.perf_counter()
    
    for num_id in range(7):
        cn_folder, cn_label = NUM_TO_CN[num_id]
        d = os.path.join(TRAIN_DIR, str(num_id))
        if not os.path.exists(d):
            print(f"  跳过 {cn_folder}({num_id}) - 目录不存在")
            continue
        
        files = sorted(glob.glob(os.path.join(d, '*.jpg')))
        label_idx = LABEL2IDX[cn_label]
        print(f"\n处理 {cn_folder}: {len(files)} 张 ...", end=" ", flush=True)
        
        t0 = time.perf_counter()
        count = 0
        fail_this = 0
        for fp in files:
            try:
                lm = extract_landmarks(landmarker, mp_obj, fp)
                if lm is None or lm.shape[0] != 468:
                    fail_this += 1
                    continue
                hm = landmarks_to_heatmap(lm)
                all_hmaps.append(hm)
                all_labels.append(label_idx)
                count += 1
            except Exception as e:
                fail_this += 1
        
        elapsed = time.perf_counter() - t0
        per_class[cn_label] = count
        speed = count / elapsed if elapsed > 0 else 0
        print(f"OK {count}张 ({elapsed:.1f}s, {speed:.1f}fps) 失败{fail_this}")
        total += count
        fail += fail_this
    
    landmarker.close()
    elapsed_total = time.perf_counter() - t0_total
    
    # 汇总
    heatmaps = np.stack(all_hmaps) if all_hmaps else np.zeros((0, 3, HEATMAP_SIZE, HEATMAP_SIZE))
    labels = np.array(all_labels, dtype=np.int64)
    
    result = {
        'heatmaps': heatmaps,
        'labels': labels,
        'label_order': LABEL_ORDER,
        'source': 'pc_mediapipe',
        'total_samples': total,
        'heatmap_size': HEATMAP_SIZE,
        'per_class': per_class,
        'failed_count': fail,
        'generation_time_sec': elapsed_total,
    }
    
    save_path = os.path.join(OUTPUT_DIR, 'cnn_train_data_pc_mediapipe.npy')
    np.save(save_path, result)
    size_mb = os.path.getsize(save_path) / (1024*1024)
    
    print(f"\n{'='*60}")
    print(f"生成完毕!")
    print(f"  成功: {total} 张 | 失败: {fail}")
    print(f"  形状: {heatmaps.shape}")
    print(f"  大小: {size_mb:.0f} MB")
    print(f"  耗时: {elapsed_total:.0f}s ({total/elapsed_total:.1f} fps)")
    print(f"  保存: {save_path}")
    print(f"  Per-class: {per_class}")
    return save_path


if __name__ == '__main__':
    main()
