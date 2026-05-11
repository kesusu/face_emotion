"""
RF重训: 清洗后数据(9244张) vs 旧数据(9906张) 对比
- 扫描当前JPG, 匹配已有热力图npy, 排除已删除样本
- 对缺失npy的JPG尝试用batch_generate_heatmaps生成(或跳过)
- RF/SVM/KNN 全套对比
"""
import os, sys, time, json, glob, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from collections import Counter

warnings.filterwarnings('ignore')

# === 路径配置 ===
BASE = r'c:\Users\ke\Desktop\嵌赛\face_emotion'
DATA_DIR = os.path.join(BASE, 'RAF-DB', 'train')
CACHE_DIR = os.path.join(DATA_DIR, 'reports', 'heatmap_cache_128')  # npy在根目录!

LABEL_ORDER = ['0惊讶', '1恐惧', '2厌恶', '3快乐', '4悲伤', '5愤怒', '6中性']
LABEL_MAP = {name: i for i, name in enumerate(LABEL_ORDER)}

print("=" * 60)
print("RF Retrain: Cleaned data vs old (9906)")
print("=" * 60)

# === Step 1: 扫描当前JPG并匹配热力图 ===
# npy命名格式: heatmap_cache_128/{类别}_{jpg名}.npy  (扁平结构!)
print("\n[Step 1] Scanning JPG + matching heatmaps...")
valid_pairs = []
missing_heatmap = []

for cls_name in LABEL_ORDER:
    cls_dir = os.path.join(DATA_DIR, cls_name)
    if not os.path.exists(cls_dir):
        print(f"  [WARN] {cls_name} dir missing")
        continue

    jpgs = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
    for jpg in sorted(jpgs):
        # npy在heatmap_cache_128/根目录下, 格式: {类别}_{jpg名}.npy
        npy_name = f"{cls_name}_{jpg}.npy"
        npy_path = os.path.join(CACHE_DIR, npy_name)
        if os.path.exists(npy_path):
            valid_pairs.append((cls_name, jpg, npy_path))
        else:
            missing_heatmap.append((cls_name, jpg))

print(f"\n  有效JPG+热力图配对: {len(valid_pairs)}")
print(f"  缺少热力图的JPG: {len(missing_heatmap)}")

if len(missing_heatmap) > 0 and len(missing_heatmap) <= 50:
    print(f"  缺失详情:")
    for c, j in missing_heatmap[:10]:
        print(f"    {c}/{j}")
    if len(missing_heatmap) > 10:
        print(f"    ... 共{len(missing_heatmap)}个")

# 统计各类数量
cls_counts = Counter(p[0] for p in valid_pairs)
print("\n  各类分布:")
for name in LABEL_ORDER:
    cnt = cls_counts.get(name, 0)
    bar = '#' * (cnt // 20)
    print(f"    {name}: {cnt:5d} 张 {bar}")

# === Step 2: 加载热力图数据 ===
print(f"\n[Step 2] 加载 {len(valid_pairs)} 个热力图...")
X_list = []
y_list = []
load_errors = 0

for idx, (cls_name, jpg, npy_path) in enumerate(valid_pairs):
    try:
        data = np.load(npy_path, allow_pickle=True)
        if data.shape == ():  # scalar object array -> dict
            hm = data.item()['heatmap']  # dict{'heatmap': array(3,128,128), 'source': str}
        else:
            hm = data
        if hm.shape == (3, 128, 128):
            X_list.append(hm.flatten())  # 49152维
            y_list.append(LABEL_MAP[cls_name])
        else:
            load_errors += 1
            if load_errors <= 3:
                print(f"    [WARN] 形状异常: {npy_path} -> {hm.shape}")
    except Exception as e:
        load_errors += 1
        if load_errors <= 3:
            print(f"    [ERR] {npy_path}: {e}")
    
    # 每1000个打印进度
    if (idx + 1) % 1000 == 0:
        print(f"    进度: {idx+1}/{len(valid_pairs)} ({(idx+1)/len(valid_pairs):.0%})")

print(f"  成功加载: {len(X_list)}")
print(f"  加载错误: {load_errors}")

if len(X_list) < 100:
    print("[ERROR] 有效样本太少, 无法继续")
    sys.exit(1)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
print(f"  数据矩阵: {X.shape}, 标签: {y.shape}")

# 释放内存
del X_list, y_list, valid_pairs

# === Step 3: 划分训练/验证集 (与之前完全相同的random_state) ===
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[Step 3] 数据划分: Train={len(X_tr)}, Val={len(X_va)}")

# 验证集各类分布
val_counts = Counter(y_va)
print(f"  验证集分布:")
for name in LABEL_ORDER:
    idx = LABEL_MAP[name]
    print(f"    {name}: {val_counts.get(idx, 0):4d}张")

del X  # 释放原始数据内存

# === Step 4: PCA降维 + 标准化 ===
print(f"\n[Step 4] PCA降维 (50成分)...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)

pca = PCA(n_components=50, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_scaled)
X_va_pca = pca.transform(X_va_scaled)

explained = sum(pca.explained_variance_ratio_)
print(f"  前50成分解释方差: {explained:.4%}")

del X_tr_scaled, X_va_scaled

# === Step 5: 训练模型 ===
print(f"\n[Step 5] 训练模型...")

results = {}

# --- Random Forest ---
t0 = time.perf_counter()
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_tr_pca, y_tr)
rf_acc = rf.score(X_va_pca, y_va)
rf_time = time.perf_counter() - t0
results['RF (200树, PCA50)'] = rf_acc
print(f"  RF(200树):     {rf_acc:.1%}  ({rf_time:.1f}s)")

# --- KNN ---
t0 = time.perf_counter()
knn = KNeighborsClassifier(n_neighbors=21, metric='cosine')
knn.fit(X_tr_pca, y_tr)
knn_acc = knn.score(X_va_pca, y_va)
knn_time = time.perf_counter() - t0
results['KNN(k=21,cos)'] = knn_acc
print(f"  KNN(k=21,cos):{knn_acc:.1%}  ({knn_time:.1f}s)")

# --- SVM RBF ---
t0 = time.perf_counter()
svm = SVC(C=5, kernel='rbf', random_state=42)
svm.fit(X_tr_pca, y_tr)
svm_acc = svm.score(X_va_pca, y_va)
svm_time = time.perf_counter() - t0
results['SVM(RBF,C=5)'] = svm_acc
print(f"  SVM(RBF,C=5):  {svm_acc:.1%}  ({svm_time:.1f}s)")

# === Step 6: 逐类准确率 (最佳模型) ===
best_model_name = max(results, key=results.get)
best_acc = results[best_model_name]

print(f"\n{'='*60}")
print(f"结果对比: 清洗后({len(X_tr)+len(X_va)}张) vs 旧数据(9906张)")
print(f"{'='*60}")

# 旧数据参考值 (来自之前实验)
old_results = {
    'RF (200树, PCA50)': 0.477,
    'KNN(k=21,cos)':   0.453,
    'SVM(RBF,C=5)':     0.446,
}

print(f"\n{'模型':25s} {'旧(9906张)':>10s} {'新(清洗后)':>12s} {'变化':>8s}")
print("-" * 58)
for name in results.keys():
    old = old_results.get(name, 0)
    new = results[name]
    delta = new - old
    arrow = "★" if delta > 0 else "" if delta == 0 else ""
    print(f"{name:25s} {old:>9.1%} {new:>11.1%} {delta:+.1%}{arrow}")

print(f"\n★ 最佳模型: {best_model_name} = {best_acc:.1%}")

# 逐类准确率 (RF)
from collections import defaultdict
rf_pred = rf.predict(X_va_pca)
per_class_correct = defaultdict(int)
per_class_total = defaultdict(int)
for true_label, pred_label in zip(y_va, rf_pred):
    per_class_total[true_label] += 1
    if true_label == pred_label:
        per_class_correct[true_label] += 1

print(f"\n逐类准确率 ({best_model_name}):")
for name in LABEL_ORDER:
    idx = LABEL_MAP[name]
    total = per_class_total.get(idx, 0)
    correct = per_class_correct.get(idx, 0)
    acc = correct / total * 100 if total > 0 else 0
    print(f"  {name}: {acc:.1f}% ({correct}/{total})")

# === Step 7: 推理速度测试 ===
print(f"\n推理速度测试 (CPU):")
sample = X_va_pca[0].reshape(1, -1)

import itertools
times_rf = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = rf.predict(sample)
    times_rf.append(time.perf_counter() - t0)
print(f"  RF: avg={np.mean(times_rf)*1000:.1f}ms")

times_knn = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = knn.predict(sample)
    times_knn.append(time.perf_counter() - t0)
print(f"  KNN: avg={np.mean(times_knn)*1000:.1f}ms")

times_svm = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = svm.predict(sample)
    times_svm.append(time.perf_counter() - t0)
print(f"  SVM: avg={np.mean(times_svm)*1000:.1f}ms")

# === 输出报告 ===
report = {
    'experiment': 'cleaned_data_v2',
    'timestamp': time.strftime('%Y-%m-%d_%H%M'),
    'total_samples': int(len(X_tr) + len(X_va)),
    'train_samples': int(len(X_tr)),
    'val_samples': int(len(X_va)),
    'removed_samples': 9906 - int(len(X_tr) + len(X_va)),
    'class_distribution': {name: cls_counts.get(name, 0) for name in LABEL_ORDER},
    'pca_components': 50,
    'pca_variance_explained': float(explained),
    'results_old': old_results,
    'results_new': {k: float(v) for k, v in results.items()},
    'best_model': best_model_name,
    'best_accuracy': float(best_acc),
    'per_class_accuracy': {
        name: round(per_class_correct.get(LABEL_MAP[name], 0) / max(per_class_total.get(LABEL_MAP[name], 1), 1), 3)
        for name in LABEL_ORDER
    },
    'inference_speed_ms': {
        'RF': round(float(np.mean(times_rf) * 1000), 2),
        'KNN': round(float(np.mean(times_knn) * 1000), 2),
        'SVM': round(float(np.mean(times_svm) * 1000), 2),
    }
}

out_path = os.path.join(BASE, 'RAF-DB', 'train', 'reports', 'rf_cleaned_v2_report.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n报告已保存: {out_path}")

# === Step 8: 保存正式版模型 (带版本号) ===
print(f"\n[Step 8] 保存正式版模型...")
import joblib

model_dir = os.path.join(BASE, 'RAF-DB', 'train', 'reports')
timestamp = time.strftime('%Y%m%d_%H%M')

# 保存SVM最佳模型
svm_pkg = {
    'model': svm,
    'pca': pca,
    'scaler': scaler,
    'label_order': LABEL_ORDER,
    'label_map': LABEL_MAP,
    'meta': {
        'version': f'v16_svm_cleaned_{best_acc:.0%}',
        'accuracy': float(best_acc),
        'n_samples': int(len(X_tr) + len(X_va)),
        'pca_components': 50,
        'pca_variance': float(explained),
        'train_date': timestamp,
        'data_source': 'cleaned_20260509',
        'per_class_acc': report['per_class_accuracy'],
    }
}
svm_path = os.path.join(model_dir, f'v16_svm_cleaned_{best_acc:.0%}_{timestamp}.joblib')
joblib.dump(svm_pkg, svm_path)
print(f"  SVM模型: {os.path.basename(svm_path)} ({os.path.getsize(svm_path)/1024:.0f}KB)")

# 保存RF模型(第二好)
rf_pkg = {
    'model': rf,
    'pca': pca,
    'scaler': scaler,
    'label_order': LABEL_ORDER,
    'label_map': LABEL_MAP,
    'meta': {
        'version': f'v16_rf_cleaned_{rf_acc:.0%}',
        'accuracy': float(rf_acc),
        'n_samples': int(len(X_tr) + len(X_va)),
        'pca_components': 50,
        'pca_variance': float(explained),
        'train_date': timestamp,
        'data_source': 'cleaned_20260509',
    }
}
rf_path = os.path.join(model_dir, f'v16_rf_cleaned_{rf_acc:.0%}_{timestamp}.joblib')
joblib.dump(rf_pkg, rf_path)
print(f"  RF模型:  {os.path.basename(rf_path)} ({os.path.getsize(rf_path)/1024:.0f}KB)")

# 保存KNN模型
knn_pkg = {
    'model': knn,
    'pca': pca,
    'scaler': scaler,
    'label_order': LABEL_ORDER,
    'label_map': LABEL_MAP,
    'meta': {
        'version': f'v16_knn_cleaned_{knn_acc:.0%}',
        'accuracy': float(knn_acc),
        'n_samples': int(len(X_tr) + len(X_va)),
        'pca_components': 50,
        'train_date': timestamp,
    }
}
knn_path = os.path.join(model_dir, f'v16_knn_cleaned_{knn_acc:.0%}_{timestamp}.joblib')
joblib.dump(knn_pkg, knn_path)
print(f"  KNN模型: {os.path.basename(knn_path)} ({os.path.getsize(knn_path)/1024:.0f}KB)")

print(f"\n全部完成! 最佳: SVM(RBF,C=5) = {best_acc:.1%}")
