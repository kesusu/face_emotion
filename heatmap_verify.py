"""
PC端 MediaPipe 热力图验证脚本
===============================
目标: 用PC的MediaPipe FaceMesh对7类各抽1张图,生成热力图,
      与板端的fiboaisdk输出做对比,验证能否替代板子跑数据

输出:
  - heatmap_verify/ 目录下每类的 jpg / npy / 可视化png
  - verify_report.txt 详细对比报告
  - landmarks坐标统计(用于判断两套引擎是否兼容)

用法: py -3.11 heatmap_verify.py
"""
import os, sys, json, time

# 强制用Python 3.11的环境
print(f"[Python] {sys.version}")

import numpy as np
import cv2

import mediapipe as mp
print(f"[MediaPipe] {mp.__version__}")

# ==================== 路径配置 ====================
WORKSPACE = r"C:\Users\ke\Desktop\嵌赛\face_emotion"
RAF_DB_TRAIN = os.path.join(WORKSPACE, "RAF-DB", "train")
OUTPUT_DIR = os.path.join(WORKSPACE, "heatmap_verify")

# 类别映射 (目录名 -> 中文名)
CATEGORIES = {
    '0': '惊讶', '1': '恐惧', '2': '厌恶',
    '3': '快乐', '4': '悲伤', '5': '愤怒', '6': '中性',
}

# 从 cnn_heatmap_classifier.py 复制过来的完全相同的热力图函数
HEATMAP_SIZE = 64
GAUSS_SIGMA = 1.5

def landmarks_to_heatmap(landmarks_468x3, size=HEATMAP_SIZE, sigma=GAUSS_SIGMA):
    """与板端 cnn_heatmap_classifier.py 完全一致的热力图生成函数"""
    lm = np.asarray(landmarks_468x3, dtype=np.float64)
    # 兼容新旧MediaPipe: 新版FaceLandmarker输出478点(含虹膜/嘴唇精细化),
    # 旧版FaceMesh输出468点. 前468点索引完全一致, 直接截断即可.
    if lm.shape[0] >= 468:
        lm = lm[:468]
    elif lm.shape[0] != 468:
        return np.zeros((3, size, size), dtype=np.float32)

    x = lm[:, 0] * (size - 1)
    y = lm[:, 1] * (size - 1)
    z = lm[:, 2]

    xx = np.arange(size, dtype=np.float64)
    yy = np.arange(size, dtype=np.float64)
    gx, gy = np.meshgrid(xx, yy)
    sigma_px = sigma * size / 10

    # Channel 0: xy位置
    h0 = np.zeros((size, size), dtype=np.float64)
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            h0 += np.exp(-d2 / (2 * sigma_px**2))

    # Channel 1: z深度加权
    h1 = np.zeros((size, size), dtype=np.float64)
    for i in range(468):
        if 0 <= x[i] < size and 0 <= y[i] < size:
            weight = (z[i] + 1) / 2
            d2 = (gx - x[i])**2 + (gy - y[i])**2
            h1 += weight * np.exp(-d2 / (2 * sigma_px**2))

    # Channel 2: 区域分组
    h2 = np.zeros((size, size), dtype=np.float64)
    REGIONS = {
        'eyebrow': list(range(70,108)) + list(range(300,338)),
        'eyes': [33,133,159,145,386,374,263,362] +
                list(range(33,134)) + list(range(362,394)),
        'nose': [1,2,98,327,168,6,197,195,5],
        'mouth': list(range(61,92)) + list(range(291,322)) +
               [13,14,17,0,37,84,87,178,409,375,409,270],
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

    # 归一化到[0,1]
    def norm(h):
        mx = h.max()
        return (h / mx).astype(np.float32) if mx > 0 else h.astype(np.float32)

    return np.stack([norm(h0), norm(h1), norm(h2)], axis=0)


def visualize_heatmap(hmap, save_path, title=""):
    """将3通道热力图保存为可视化PNG"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=14)

    # RGB合成
    rgb = np.stack([hmap[0], hmap[1], hmap[2]], axis=-1)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB合成')
    axes[0].axis('off')

    # 各通道
    ch_names = ['Ch0: xy位置', 'Ch1: z深度', 'Ch2: 区域分组']
    for c in range(3):
        axes[c+1].imshow(hmap[c], cmap='hot', vmin=0, vmax=1)
        axes[c+1].set_title(ch_names[c])
        axes[c+1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def extract_landmarks_mediapipe(image_path):
    """
    用MediaPipe FaceLandmarker (新tasks API) 提取468个3D关键点
    返回: landmarks (468, 3) 归一化坐标 [0,1], 或 None(检测失败)
    
    注意: MediaPipe 0.10.x 使用新的 mp.tasks API (非旧版 mp.solutions)
    """
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

    model_path = os.path.expanduser('~/.mediapipe/models/face_landmarker.task')
    if not os.path.exists(model_path):
        print(f"    [ERROR] 模型文件不存在: {model_path}")
        return None

    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,   # 不需要blendshape
        output_facial_transformation_matrixes=False,  # 不需要变换矩阵
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # cv2.imread 不支持中文路径, 用 np.fromfile + imdecode 替代
        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            print(f"    [ERROR] 无法读取图片: {image_path}")
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            print(f"    [WARN] 未检测到人脸: {image_path}")
            return None

        landmark = results.face_landmarks[0]

        # 提取468点的归一化坐标
        points = []
        for lm in landmark:
            points.append([lm.x, lm.y, lm.z])

        arr = np.array(points, dtype=np.float32)  # (468, 3)
        return arr


# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PC(MediaPipe) vs 板端(fiboaisdk) 热力图验证报告")
    report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")

    all_results = {}  # 存储所有类别的结果供汇总

    for item in sorted(os.listdir(RAF_DB_TRAIN)):
        cls_dir = os.path.join(RAF_DB_TRAIN, item)
        if not os.path.isdir(cls_dir):
            continue

        # 从目录名中提取类别ID (支持 "0" / "0惊讶" 两种命名)
        cls_id = item[0]  # 取首字符作为ID
        if cls_id not in CATEGORIES:
            continue
        cls_name = CATEGORIES[cls_id]

        # 取该类别第1张jpg作为样本
        jpgs = sorted([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        if not jpgs:
            report_lines.append(f"[{cls_name}] 无jpg文件")
            continue

        sample_jpg = jpgs[0]
        sample_path = os.path.join(cls_dir, sample_jpg)

        # 创建该类别的输出子目录
        cls_out = os.path.join(OUTPUT_DIR, f"{cls_id}_{cls_name}")
        os.makedirs(cls_out, exist_ok=True)

        # 复制原图过去(方便对照)
        orig_copy = os.path.join(cls_out, f"original_{sample_jpg}")
        import shutil
        shutil.copy2(sample_path, orig_copy)

        print(f"\n[{cls_name}] 处理: {sample_path}")

        # ===== Step 1: MediaPipe提取468点 =====
        t0 = time.time()
        lm = extract_landmarks_mediapipe(sample_path)
        t_extract = time.time() - t0

        if lm is None:
            report_lines.append(f"\n--- [{cls_id} {cls_name}] FAILED ---")
            report_lines.append(f"  原图: {sample_path}")
            report_lines.append(f"  结果: 人脸检测失败!")
            continue

        print(f"  [OK] 468点提取完成 ({t_extract:.3f}s)")
        print(f"    形状: {lm.shape}, 范围: x=[{lm[:,0].min():.3f},{lm[:,0].max():.3f}], y=[{lm[:,1].min():.3f},{lm[:,1].max():.3f}], z=[{lm[:,2].min():.3f},{lm[:,2].max():.3f}]")

        # ===== Step 2: 生成热力图 (使用与板端完全相同的函数) =====
        t1 = time.time()
        hmap = landmarks_to_heatmap(lm, size=HEATMAP_SIZE, sigma=GAUSS_SIGMA)
        t_heatmap = time.time() - t1

        print(f"  [OK] 热力图生成完成 ({t_heatmap:.3f}s), shape={hmap.shape}")
        print(f"    Ch0范围: [{hmap[0].min():.4f}, {hmap[0].max():.4f}]")
        print(f"    Ch1范围: [{hmap[1].min():.4f}, {hmap[1].max():.4f}]")
        print(f"    Ch2范围: [{hmap[2].min():.4f}, {hmap[2].max():.4f}]")

        # ===== Step 3: 保存文件 =====
        npy_path = os.path.join(cls_out, "heatmap_PC_MediaPipe.npy")
        np.save(npy_path, hmap)

        # 保存landmarks原始坐标(供板端AI对比)
        lm_path = os.path.join(cls_out, "landmarks_468x3_PC.npy")
        np.save(lm_path, lm)

        # 保存landmarks文本摘要
        lm_txt = os.path.join(cls_out, "landmarks_stats.txt")
        with open(lm_txt, 'w', encoding='utf-8') as f:
            f.write(f"来源: PC端 MediaPipe FaceMesh (refine_landmarks=True)\n")
            f.write(f"图片: {sample_path}\n")
            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"形状: {lm.shape}\n\n")
            f.write(f"x轴: min={lm[:,0].min():.6f} max={lm[:,0].max():.6f} mean={lm[:,0].mean():.6f}\n")
            f.write(f"y轴: min={lm[:,1].min():.6f} max={lm[:,1].max():.6f} mean={lm[:,1].mean():.6f}\n")
            f.write(f"z轴: min={lm[:,2].min():.6f} max={lm[:,2].max():.6f} mean={lm[:,2].mean():.6f}\n\n")
            f.write("前20个点坐标:\n")
            for i in range(min(20, 468)):
                f.write(f"  point[{i:3d}]: x={lm[i,0]:.6f}  y={lm[i,1]:.6f}  z={lm[i,2]:.6f}\n")
            f.write("\n...\n")
            f.write(f"最后5个点坐标:\n")
            for i in range(463, 468):
                f.write(f"  point[{i:3d}]: x={lm[i,0]:.6f}  y={lm[i,1]:.6f}  z={lm[i,2]:.6f}\n")

        # 尝试可视化
        try:
            vis_path = os.path.join(cls_out, "heatmap_visualization.png")
            visualize_heatmap(hmap, vis_path, title=f"{cls_name} (PC MediaPipe)")
            print(f"  [OK] 可视化已保存")
        except ImportError:
            print(f"  [WARN] matplotlib未安装,跳过可视化")

        # ===== 汇总信息 =====
        result = {
            'class_id': cls_id,
            'class_name': cls_name,
            'source_image': sample_path,
            'image_size': '',  # 稍后填入
            'extract_time_s': round(t_extract, 4),
            'heatmap_shape': str(hmap.shape),
            'heatmap_npy': npy_path,
            'landmarks_npy': lm_path,
            'landmarks_x_range': [round(lm[:,0].min(), 6), round(lm[:,0].max(), 6)],
            'landmarks_y_range': [round(lm[:,1].min(), 6), round(lm[:,1].max(), 6)],
            'landmarks_z_range': [round(lm[:,2].min(), 6), round(lm[:,2].max(), 6)],
            'ch0_range': [round(float(hmap[0].min()), 4), round(float(hmap[0].max()), 4)],
            'ch1_range': [round(float(hmap[1].min()), 4), round(float(hmap[1].max()), 4)],
            'ch2_range': [round(float(hmap[2].min()), 4), round(float(hmap[2].max()), 4)],
            # 需要板端填充的数据:
            'board_landmarks_npy': '',
            'board_heatmap_npy': '',
            'cosine_similarity_ch0': None,
            'cosine_similarity_ch1': None,
            'cosine_similarity_ch2': None,
            'compatible': False,
        }

        # 读图片尺寸
        img = cv2.imread(sample_path)
        if img is not None:
            result['image_size'] = f"{img.shape[1]}x{img.shape[0]}"

        all_results[f"{cls_id}_{cls_name}"] = result

        # 写报告段落
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"[{cls_id}] {cls_name}")
        report_lines.append(f"{'='*60}")
        report_lines.append(f"  原始图片: {sample_path}")
        report_lines.append(f"  图片尺寸: {result['image_size']}")
        report_lines.append(f"  PC提取耗时: {t_extract:.3f}s | 热力图耗时: {t_heatmap:.3f}s")
        report_lines.append(f"  Landmarks形状: {lm.shape}")
        report_lines.append(f"  X范围: [{lm[:,0].min():.4f}, {lm[:,0].max():.4f}]")
        report_lines.append(f"  Y范围: [{lm[:,1].min():.4f}, {lm[:,1].max():.4f}]")
        report_lines.append(f"  Z范围: [{lm[:,2].min():.4f}, {lm[:,2].max():.4f}]")
        report_lines.append(f"  热力图: {npy_path}")
        report_lines.append(f"  热力图形状: {hmap.shape}")
        report_lines.append(f"  输出目录: {cls_out}/")

    # ===== 最终报告 =====
    report_lines.append(f"\n{'='*80}")
    report_lines.append("汇总")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"成功处理: {len(all_results)}/7 类别")
    report_lines.append(f"输出根目录: {OUTPUT_DIR}/")
    report_lines.append("")
    report_lines.append("每个类别输出文件:")
    for name, r in all_results.items():
        report_lines.append(f"  [{name}]")
        report_lines.append(f"    原图: {r['source_image']}")
        report_lines.append(f"    PC landmarks: {r['landmarks_npy']}")
        report_lines.append(f"    PC heatmap:   {r['heatmap_npy']}")

    report_lines.append(f"\n{'='*80}")
    report_lines.append("待板端AI填写: 对比验证")
    report_lines.append(f"{'='*80}")
    report_lines.append("""
请对以上每张图片执行以下操作:

1. 用板端的 fiboaisdk/FaceEmotionDetector 对同一张原图做推理:
   python run_face_keypoint.py --image <原图路径> --save-landmarks <输出npy路径>

2. 用相同的 landmarks_to_heatmap() 函数生成热力图:
   from cnn_heatmap_classifier import landmarks_to_heatmap
   board_lm = np.load('<板端landmarks.npy>')
   board_hmap = landmarks_to_heatmap(board_lm)
   np.save('<板端heatmap.npy>', board_hmap)

3. 对比两套landmarks:
   pc_lm = np.load('landmarks_468x3_PC.npy')
   board_lm = np.load('板端landmarks.npy')
   
   a) 形状是否相同? 都应该是(468, 3)?
   b) 坐标范围是否接近? 
      PC:  x=[?, ?]  y=[?, ?]  z=[?, ?]
      板端: x=[?, ?]  y=[?, ?]  z=[?, ?]
   c) 余弦相似度? 
      cos_sim = np.dot(pc_lm.flatten(), board_lm.flatten()) / 
                 (np.linalg.norm(pc_lm.flatten()) * np.linalg.norm(board_lm.flatten()))

4. 对比两张热力图的余弦相似度(每个channel分别算):
   for ch in range(3):
       cos = np.dot(pc_hmap[ch].flatten(), board_hmap[ch].flatten()) / 
              (np.linalg.norm(pc_hmap[ch].flatten()) * np.linalg.norm(board_hmap[ch].flatten()))
       print(f'Ch{ch}: cos_sim = {cos:.6f}')

判定标准:
  - 余弦相似度 > 0.95: 完全兼容, PC可以替代板端生成热力图
  - 余弦相似度 0.85-0.95: 基本兼容, 可能有微小偏移但可接受
  - 余弦相似度 < 0.85: 不兼容, 需要进一步分析差异原因
""")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "verify_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n[OK] 报告已保存: {report_path}")

    # 保存结构化结果JSON(方便程序读取)
    json_path = os.path.join(OUTPUT_DIR, "verify_results.json")
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON结果已保存: {json_path}")

    print(f"\n全部完成! 请查看: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
