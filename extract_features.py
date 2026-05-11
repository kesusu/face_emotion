"""
人脸情绪特征提取脚本
基于 MediaPipe Face Mesh 468关键点
与开发板 mediapipe-face SNPE 模型输出格式完全一致

用法: python extract_features.py [图片路径]
默认: test_0010.jpg
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os


def extract_face_landmarks(image_path):
    """使用 MediaPipe Face Mesh 提取468个面部关键点"""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # 包含虹膜点(468-477)，与开发板一致
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None, None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("错误: 未检测到人脸")
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # 转换为 numpy 数组 (478, 3) — 含虹膜
    pts = np.array([[p.x, p.y, p.z] for p in landmarks])
    face_mesh.close()

    return pts, img


def compute_emotion_features(pts):
    """
    根据文档中的8个特征公式计算面部表情特征
    坐标为归一化坐标(0~1)
    """
    features = {}

    # ────────────────────────────────────────────
    # 特征1: 眉毛高度 (Eyebrow Height)
    # 左眉高度 = 点105.y - 点33.y (眉中点相对于左眼外角)
    # 右眉高度 = 点334.y - 点263.y
    # ────────────────────────────────────────────
    left_brow_h = pts[105][1] - pts[33][1]
    right_brow_h = pts[334][1] - pts[263][1]
    features['eyebrow_height'] = (left_brow_h + right_brow_h) / 2
    features['left_brow_height'] = left_brow_h
    features['right_brow_height'] = right_brow_h

    # ────────────────────────────────────────────
    # 特征2: 眉间距离 (Inter-brow Distance)
    # 眉间距离 = |点70.x - 点300.x|
    # ────────────────────────────────────────────
    features['inter_brow_distance'] = abs(pts[70][0] - pts[300][0])

    # ────────────────────────────────────────────
    # 特征3: 眼睛开合度 (Eye Openness) ★最重要
    # 左眼开合度 = |点159.y - 点145.y| / |点33.x - 点133.x|
    # 右眼开合度 = |点386.y - 点374.y| / |点263.x - 点362.x|
    # ────────────────────────────────────────────
    left_eye_open = abs(pts[159][1] - pts[145][1]) / abs(pts[33][0] - pts[133][0])
    right_eye_open = abs(pts[386][1] - pts[374][1]) / abs(pts[263][0] - pts[362][0])
    features['eye_openness'] = (left_eye_open + right_eye_open) / 2
    features['left_eye_openness'] = left_eye_open
    features['right_eye_openness'] = right_eye_open

    # ────────────────────────────────────────────
    # 特征4: 嘴角高度 (Mouth Corner Height) ★最重要
    # 参考线 = (点0.y + 点17.y) / 2
    # 左嘴角偏移 = 点61.y - 参考线 (正=下垂, 负=上扬)
    # 右嘴角偏移 = 点291.y - 参考线
    # ────────────────────────────────────────────
    ref_line = (pts[0][1] + pts[17][1]) / 2
    left_mouth_offset = pts[61][1] - ref_line
    right_mouth_offset = pts[291][1] - ref_line
    features['mouth_corner_height'] = (left_mouth_offset + right_mouth_offset) / 2
    features['left_mouth_corner_offset'] = left_mouth_offset
    features['right_mouth_corner_offset'] = right_mouth_offset

    # ────────────────────────────────────────────
    # 特征5: 嘴巴开合度 (Mouth Openness)
    # 嘴巴开合度 = |点13.y - 点14.y| / |点61.x - 点291.x|
    # ────────────────────────────────────────────
    mouth_open = abs(pts[13][1] - pts[14][1]) / abs(pts[61][0] - pts[291][0])
    features['mouth_openness'] = mouth_open

    # ────────────────────────────────────────────
    # 特征6: 鼻翼扩张 (Nose Wing Expansion)
    # 鼻翼宽度 = |点98.x - 点327.x|
    # 鼻梁宽度 = |点48.x - 点278.x|
    # 鼻翼比 = 鼻翼宽度 / 鼻梁宽度
    # ────────────────────────────────────────────
    nose_wing_w = abs(pts[98][0] - pts[327][0])
    nose_bridge_w = abs(pts[48][0] - pts[278][0])
    features['nose_wing_ratio'] = nose_wing_w / nose_bridge_w if nose_bridge_w > 0 else 0
    features['nose_wing_width'] = nose_wing_w
    features['nose_bridge_width'] = nose_bridge_w

    # ────────────────────────────────────────────
    # 特征7: 下巴下垂 (Jaw Drop)
    # 下巴下垂度 = (点152.y - 点17.y) / |点10.y - 点152.y|
    # ────────────────────────────────────────────
    face_h = abs(pts[10][1] - pts[152][1])
    features['jaw_drop'] = (pts[152][1] - pts[17][1]) / face_h if face_h > 0 else 0

    # ────────────────────────────────────────────
    # 特征8: 眉毛内端上抬 (Inner Brow Raise)
    # 左眉内端高度 = 点70.y - 点33.y
    # 右眉内端高度 = 点300.y - 点263.y
    # 眉内端抬升 = avg(左,右) - 眉毛高度
    # ────────────────────────────────────────────
    left_inner_brow = pts[70][1] - pts[33][1]
    right_inner_brow = pts[300][1] - pts[263][1]
    features['inner_brow_raise'] = (left_inner_brow + right_inner_brow) / 2 - features['eyebrow_height']

    return features


def classify_emotion(features):
    """
    基于特征值简单规则判断情绪（参考文档第五节）
    注意: 阈值需根据实际数据微调
    """
    eb_h = features['eyebrow_height']       # 眉毛高度（负值=高）
    ib_dist = features['inter_brow_distance']  # 眉间距离
    eye_open = features['eye_openness']      # 眼睛开合度
    mouth_h = features['mouth_corner_height']  # 嘴角高度（负=上扬）
    mouth_open = features['mouth_openness']   # 嘴巴开合度
    nose_r = features['nose_wing_ratio']      # 鼻翼比
    jaw = features['jaw_drop']               # 下巴下垂
    inner_raise = features['inner_brow_raise']  # 眉内端抬升

    # 简单规则判断
    scores = {
        'Happy': 0.0,
        'Sad': 0.0,
        'Angry': 0.0,
        'Surprise': 0.0,
        'Fear': 0.0,
        'Disgust': 0.0,
        'Neutral': 0.0,
    }

    # ── 开心 Happy ──
    if mouth_h < -0.005:  # 嘴角上扬
        scores['Happy'] += 3.0
    if mouth_h < -0.01:
        scores['Happy'] += 1.5
    if eye_open < 0.25 and mouth_h < -0.005:  # 笑眼
        scores['Happy'] += 1.5
    if mouth_open < 0.15:  # 嘴巴不太大张
        scores['Happy'] += 0.5

    # ── 悲伤 Sad ──
    if mouth_h > 0.005:  # 嘴角下垂
        scores['Sad'] += 2.0
    if inner_raise < -0.005:  # 八字眉：内端比整体更高（y更小=更负）
        scores['Sad'] += 2.0
    if eye_open < 0.2:  # 眼睛偏小
        scores['Sad'] += 1.0

    # ── 愤怒 Angry ──
    if eb_h > 0.01:  # 眉毛低（y值大）
        scores['Angry'] += 2.0
    if ib_dist < 0.12:  # 眉间距离小
        scores['Angry'] += 2.0
    if nose_r > 1.8:  # 鼻翼扩张
        scores['Angry'] += 1.0
    if eye_open > 0.3:  # 瞪眼
        scores['Angry'] += 1.0

    # ── 惊讶 Surprise ──
    if eb_h < -0.02:  # 眉毛大幅上扬
        scores['Surprise'] += 2.5
    if mouth_open > 0.3:  # 嘴大张
        scores['Surprise'] += 2.5
    if eye_open > 0.35:  # 眼睛圆睁
        scores['Surprise'] += 1.5
    if jaw > 0.35:  # 下巴下移
        scores['Surprise'] += 1.0

    # ── 恐惧 Fear ──
    if eb_h < -0.01 and inner_raise < -0.003:  # 眉毛整体上扬+内端上抬
        scores['Fear'] += 2.0
    if eye_open > 0.3 and mouth_open > 0.1 and mouth_open < 0.3:
        scores['Fear'] += 2.0
    if mouth_h > -0.003 and mouth_h < 0.005:  # 嘴角紧张
        scores['Fear'] += 1.0

    # ── 厌恶 Disgust ──
    if nose_r > 1.8:  # 鼻翼扩张
        scores['Disgust'] += 2.0
    if mouth_open > 0.1 and abs(mouth_h) < 0.005:  # 上唇上抬+嘴角不对称
        scores['Disgust'] += 1.0
    if eb_h > 0.005:  # 皱眉（轻微）
        scores['Disgust'] += 1.0

    # ── 中性 Neutral ──
    if abs(mouth_h) < 0.003 and abs(eb_h) < 0.01 and 0.15 < eye_open < 0.3:
        scores['Neutral'] += 3.0

    # 排序
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions


def draw_landmarks_on_image(img, pts, key_points_only=True):
    """在图片上绘制关键点"""
    h, w = img.shape[:2]
    result = img.copy()

    # 核心情绪识别关键点（文档第六节速查表）
    emotion_key_indices = [
        70, 105, 107, 300, 334, 336,  # 眉毛
        159, 145, 33, 133,             # 左眼
        386, 374, 263, 362,            # 右眼
        61, 291, 13, 14, 0, 17,        # 嘴巴
        1, 98, 327, 152, 10,           # 辅助点
    ]

    if key_points_only:
        indices = emotion_key_indices
    else:
        indices = range(min(len(pts), 468))

    for i in indices:
        if i < len(pts):
            x = int(pts[i][0] * w)
            y = int(pts[i][1] * h)
            color = (0, 255, 0) if i in emotion_key_indices else (0, 0, 255)
            radius = 3 if i in emotion_key_indices else 1
            cv2.circle(result, (x, y), radius, color, -1)
            if i in emotion_key_indices and key_points_only:
                cv2.putText(result, str(i), (x + 3, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return result


def main():
    # 默认图片路径
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_0010.jpg")
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    print("=" * 70)
    print("  MediaPipe Face Mesh 面部表情特征提取")
    print("  模型: 468关键点 + 虹膜(refine_landmarks=True)")
    print("=" * 70)
    print(f"\n图片: {image_path}")

    # 1. 提取关键点
    pts, img = extract_face_landmarks(image_path)
    if pts is None:
        sys.exit(1)

    print(f"检测到人脸! 关键点数量: {len(pts)} (含虹膜点)")
    h, w = img.shape[:2]

    # 2. 打印核心关键点坐标
    print("\n" + "─" * 60)
    print("  核心情绪识别关键点坐标 (归一化 0~1):")
    print("─" * 60)
    key_indices = [70, 105, 107, 300, 334, 336,
                   159, 145, 33, 133,
                   386, 374, 263, 362,
                   61, 291, 13, 14, 0, 17,
                   1, 98, 327, 152, 10]
    key_names = {
        70: "左眉内端", 105: "左眉中点", 107: "左眉外端",
        300: "右眉内端", 334: "右眉中点", 336: "右眉外端",
        159: "左眼上眼睑", 145: "左眼下眼睑", 33: "左眼外角", 133: "左眼内角",
        386: "右眼上眼睑", 374: "右眼下眼睑", 263: "右眼外角", 362: "右眼内角",
        61: "左嘴角", 291: "右嘴角", 13: "上唇中心", 14: "下唇中心",
        0: "上唇线中点", 17: "下唇线中点",
        1: "鼻尖", 98: "左鼻翼", 327: "右鼻翼", 152: "下巴最低点", 10: "额头顶部",
    }
    print(f"{'点号':>4}  {'位置':<12}  {'x':>8}  {'y':>8}  {'z':>8}")
    print("-" * 60)
    for idx in key_indices:
        name = key_names.get(idx, "")
        print(f"{idx:>4}  {name:<12}  {pts[idx][0]:>8.4f}  {pts[idx][1]:>8.4f}  {pts[idx][2]:>8.4f}")

    # 3. 计算8个特征
    features = compute_emotion_features(pts)
    print("\n" + "=" * 60)
    print("  8个情绪特征计算结果:")
    print("=" * 60)

    feature_desc = {
        'eyebrow_height':      ('特征1: 眉毛高度',       '负值=高(惊讶/恐惧), 正值=低(愤怒)'),
        'inter_brow_distance':  ('特征2: 眉间距离',       '小=皱眉(愤怒), 大=舒展(开心)'),
        'eye_openness':         ('特征3: 眼睛开合度 ★',   '大=圆睁(惊讶), 小=眯眼(开心/悲伤)'),
        'mouth_corner_height':  ('特征4: 嘴角高度 ★',     '负=上扬(开心), 正=下垂(悲伤)'),
        'mouth_openness':       ('特征5: 嘴巴开合度',     '大=惊讶, 小=闭合'),
        'nose_wing_ratio':      ('特征6: 鼻翼扩张比',     '大=厌恶/愤怒'),
        'jaw_drop':             ('特征7: 下巴下垂度',     '大=惊讶'),
        'inner_brow_raise':     ('特征8: 眉内端抬升',     '负=八字眉(悲伤), 正=皱眉(愤怒)'),
    }

    for key, (label, desc) in feature_desc.items():
        val = features[key]
        print(f"  {label:<22} = {val:>8.4f}   ← {desc}")

    # 4. 情绪判断
    emotions = classify_emotion(features)
    print("\n" + "=" * 60)
    print("  情绪判断结果 (规则匹配得分):")
    print("=" * 60)
    for emotion, score in emotions:
        bar = "█" * int(score * 3) if score > 0 else ""
        print(f"  {emotion:<10}  {score:>5.1f}  {bar}")

    top_emotion = emotions[0][0]
    top_score = emotions[0][1]
    if top_score < 1.0:
        top_emotion = "Neutral"
    print(f"\n  >>> 判定情绪: {top_emotion} (得分: {top_score:.1f})")

    # 5. 保存标注图片
    result_img = draw_landmarks_on_image(img, pts, key_points_only=True)
    output_path = os.path.splitext(image_path)[0] + "_landmarks.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"\n  关键点标注图已保存: {output_path}")


if __name__ == "__main__":
    main()
