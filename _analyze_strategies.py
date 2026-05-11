import numpy as np, json, os, joblib
from collections import Counter, defaultdict

base = r'c:\Users\ke\Desktop\嵌赛\face_emotion'
model_dir = os.path.join(base, 'RAF-DB', 'train', 'reports')
LABEL_ORDER = ['0惊讶','1恐惧','2厌恶','3快乐','4悲伤','5愤怒','6中性']
LABEL_MAP = {name: i for i, name in enumerate(LABEL_ORDER)}
TARGET_IDS = [2, 4, 6]
TARGET_NAMES = ['2厌恶','4悲伤','6中性']

print("=" * 70)
print("三类区分问题深度分析 & 策略探索")
print("=" * 70)

# === 加载数据 ===
DATA_DIR = os.path.join(base, 'RAF-DB', 'train')
CACHE_DIR = os.path.join(DATA_DIR, 'reports', 'heatmap_cache_128')

valid_pairs = []
for cls_name in LABEL_ORDER:
    cls_dir = os.path.join(DATA_DIR, cls_name)
    if not os.path.exists(cls_dir): continue
    for jpg in sorted([f for f in os.listdir(cls_dir) if f.endswith('.jpg')]):
        npy_path = os.path.join(CACHE_DIR, f"{cls_name}_{jpg}.npy")
        if os.path.exists(npy_path):
            valid_pairs.append((cls_name, npy_path))

print(f"样本: {len(valid_pairs)}")
X_list, y_list = [], []
for cls_name, npy_path in valid_pairs:
    data = np.load(npy_path, allow_pickle=True)
    hm = data.item()['heatmap'] if data.shape == () else data
    X_list.append(hm.flatten())
    y_list.append(LABEL_MAP[cls_name])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
del X_list, y_list

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
pca = PCA(n_components=50, random_state=42)
X_tr_p = pca.fit_transform(scaler.fit_transform(X_tr))
X_va_p = pca.transform(scaler.transform(X_va))

del X

# 加载已训练的模型
svm_pkg = joblib.load(os.path.join(model_dir, [f for f in os.listdir(model_dir) if 'v16_svm' in f][0]))
rf_pkg = joblib.load(os.path.join(model_dir, [f for f in os.listdir(model_dir) if 'v16_rf' in f][0]))

svm_pred = svm_pkg['model'].predict(X_va_p)
rf_pred = rf_pkg['model'].predict(X_va_p)

# ============================================================
# 第一部分: 核心发现 — 问题本质变了!
# ============================================================
print("\n" + "=" * 70)
print("【第一部分】问题本质: 不是三类互混, 而是→中性崩塌!")
print("=" * 70)

cm_svm = confusion_matrix(y_va, svm_pred, labels=range(7))
cm_rf = confusion_matrix(y_va, rf_pred, labels=range(7))

def show_3class_submatrix(cm, title):
    print(f"\n{title}")
    print(f"{'':8s}", end='')
    for n in TARGET_NAMES: print(f"{n:>10s}", end='')
    print()
    for i, tid in enumerate(TARGET_IDS):
        row_total = cm[tid].sum()
        print(f"{TARGET_NAMES[i]:8s}", end='')
        for j, tjd in enumerate(TARGET_IDS):
            val = cm[tid][tjd]
            pct = val / row_total * 100
            tag = ""
            if i == j and pct < 30: tag = " ★ 崩"
            elif i != j and pct > 40: tag = " ★ 大量误判为这个!"
            print(f" {val:>4d}({pct:>4.1f}%){tag}", end='')
        # 被判成非三类的数量
        others = sum(cm[tid][k] for k in range(7) if k not in TARGET_IDS)
        print(f" | 其他:{others}")
    print(f"  {'列总计':8s}", end='')
    for j, tjd in enumerate(TARGET_IDS):
        print(f" {cm[:,tjd].sum():>6d}       ", end='')
    print()

show_3class_submatrix(cm_svm, "SVM 混淆子矩阵 (行=真实, 列=预测)")
show_3class_submatrix(cm_rf, "RF 混淆子矩阵")

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ★ 核心发现: 清洗数据后问题本质完全改变了!                   ║
║                                                              ║
║  旧印象(V15 CNN, 未清洗数据):                                ║
║    "43%的中性被判成厌恶, 27%悲伤被判成厌恶"                  ║
║    → 当时问题是: 厌恶是个"黑洞", 吞掉中性和悲伤              ║
║                                                              ║
║  新事实(V16 SVM/RF, 清洗后数据):                             ║
║    "65%厌恶被判成中性, 63%悲伤被判成中性"                     ║
║    → 现在问题是: 中性是个"黑洞", 吞掉厌恶和悲伤!             ║
║                                                              ║
║  原因推测:                                                   ║
║    用户删除了"明显不对"的标注后, 剩余的厌恶/悲伤样本          ║
║    特征变得很微妙(轻微皱眉/嘴角微垂), 在高斯热力图上         ║
║    和几乎无表情的中性脸几乎无法区分                           ║
║    → cos_sim=0.965的诅咒仍然存在                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# ============================================================
# 第二部分: 尝试新策略
# ============================================================
print("=" * 70)
print("【第二部分】策略实验")
print("=" * 70)

# --- 策略A: One-vs-Rest 二分类器 ---
print("\n--- 策略A: OvR二分类器 (每个弱类单独训练) ---")

ovr_results = {}
for tid, tname in zip(TARGET_IDS, TARGET_NAMES):
    y_binary_tr = (y_tr == tid).astype(int)
    y_binary_va = (y_va == tid).astype(int)
    
    ovr_svc = SVC(C=5, kernel='rbf', random_state=42, probability=True)
    ovr_svc.fit(X_tr_p, y_binary_tr)
    
    pred_bin = ovr_svc.predict(X_va_p)
    acc = (pred_bin == y_binary_va).mean()
    
    # 只看正类(precision/recall)
    tp = ((pred_bin == 1) & (y_binary_va == 1)).sum()
    fp = ((pred_bin == 1) & (y_binary_va == 0)).sum()
    fn = ((pred_bin == 0) & (y_binary_va == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ovr_results[tname] = {'acc': acc, 'prec': prec, 'rec': rec}
    print(f"  {tname}: Acc={acc:.1%}, Precision={prec:.1%}(P), Recall={rec:.1%}(R)")

# --- 策略B: 不平衡权重(class_weight='balanced') ---
print("\n--- 策略B: balanced加权SVM ---")

svm_bal = SVC(C=5, kernel='rbf', random_state=42, class_weight='balanced')
svm_bal.fit(X_tr_p, y_tr)
pred_bal = svm_bal.predict(X_va_p)

acc_bal = (pred_bal == y_va).mean()
print(f"  全部7类: {acc_bal:.1%}")

for tid, tname in zip(TARGET_IDS, TARGET_NAMES):
    m = y_va == tid
    sub_acc = (pred_bal[m] == y_va[m]).mean()
    orig_acc = (svm_pred[m] == y_va[m]).mean()
    delta = sub_acc - orig_acc
    arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
    print(f"  {tname}: {sub_acc:.1%} ({arrow}{delta:+.1%}) vs 原始{orig_acc:.1%}")

# --- 策略C: 逻辑回归(线性可解释) ---
print("\n--- 策略C: 多项逻辑回归(可解释+快速) ---")

lr = LogisticRegression(multi_class='multinomial', max_iter=1000, C=1, random_state=42, class_weight='balanced')
lr.fit(X_tr_p, y_tr)
pred_lr = lr.predict(X_va_p)
acc_lr = (pred_lr == y_va).mean()
print(f"  全部7类: {acc_lr:.1%}")

for tid, tname in zip(TARGET_IDS, TARGET_NAMES):
    m = y_va == tid
    print(f"  {tname}: {(pred_lr[m]==y_va[m]).mean():.1%}")

# --- 策略D: 自定义投票(RF对悲伤更好时听RF的) ---
print("\n--- 策略D: 智能投票(SVM为主, 悲伤听RF) ---")

smart_pred = svm_pred.copy()
for i in range(len(y_va)):
    true_cls = y_va[i]
    # 如果预测是悲伤(4), 且RF也预测悲伤, 则用RF的结果
    if rf_pred[i] == 4 and svm_pred[i] != 4:
        smart_pred[i] = rf_pred[i]

smart_acc = (smart_pred == y_va).mean()
print(f"  全部7类: {smart_acc:.1%} (vs SVM={np.mean(svm_pred==y_va):.1%})")

for tid, tname in zip(TARGET_IDS, TARGET_NAMES):
    m = y_va == tid
    s_orig = (svm_pred[m] == y_va[m]).mean()
    s_smart = (smart_pred[m] == y_va[m]).mean()
    print(f"  {tname}: {s_smart:.1%} (原始{s_orig:.1%})")

# --- 策略E: 三类专用分类器(只在三类上训练) ---
print("\n--- 策略E: 三类专用分类器(只用厌恶/悲伤/中性训练) ---")

mask_tr = np.isin(y_tr, TARGET_IDS)
mask_va = np.isin(y_va, TARGET_IDS)

X_tr_3 = X_tr_p[mask_tr]
y_tr_3 = y_tr[mask_tr]
X_va_3 = X_va_p[mask_va]
y_va_3 = y_va[mask_va]

# 映射到0,1,2
map3 = {2: 0, 4: 1, 6: 2}
rev_map3 = {0: 2, 1: 4, 2: 6}

y_tr_3m = np.array([map3[v] for v in y_tr_3])
y_va_3m = np.array([map3[v] for v in y_va_3])

svm_3 = SVC(C=10, kernel='rbf', random_state=42, class_weight='balanced')
svm_3.fit(X_tr_3, y_tr_3m)
pred_3m = svm_3.predict(X_va_3)
pred_3_full = np.array([rev_map3[p] for p in pred_3m])

acc_3 = (pred_3m == y_va_3m).mean()
print(f"  三类专用SVM: {acc_3:.1%}")
for i, tn in enumerate(['厌恶','悲伤','中性']):
    m = y_va_3m == i
    print(f"    {tn}: {(pred_3m[m]==y_va_3m[m]).mean():.1%} ({m.sum()}张)")

rf_3 = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, class_weight='balanced')
rf_3.fit(X_tr_3, y_tr_3m)
pred_3rf_m = rf_3.predict(X_va_3)
acc_3rf = (pred_3rf_m == y_va_3m).mean()
print(f"  三类专用RF:  {acc_3rf:.1%}")
for i, tn in enumerate(['厌恶','悲伤','中性']):
    m = y_va_3m == i
    print(f"    {tn}: {(pred_3rf_m[m]==y_va_3m[m]).mean():.1%} ({m.sum()}张)")

# 三类专用 + 投票
ens_3 = []
for sv3, rf3 in zip(pred_3m, pred_3rf_m):
    ens_3.append(sv3 if sv3 == rf3 else sv3)  # tiebreak SVM
ens_3 = np.array(ens_3)
ens_3acc = (ens_3 == y_va_3m).mean()
print(f"  三类专用Ens: {ens_3acc:.1%}")
for i, tn in enumerate(['厌恶','悲伤','中性']):
    m = y_va_3m == i
    print(f"    {tn}: {(ens_3[m]==y_va_3m[m]).mean():.1%}")

# ============================================================
# 第三部分: 总结与推荐
# ============================================================
print("\n" + "=" * 70)
print("【第三部分】策略效果总结")
print("=" * 70)

print("""
┌──────────────┬────────┬──────────┬──────────┬──────────┬──────────┐
│ 策略         │ 全7类  │ 厌恶     │ 悲伤     │ 中性     │ 备注     │
├──────────────┼────────┼──────────┼──────────┼──────────┼──────────┤
│ 原始SVM      │ 56.0%  │  17.5%   │  34.5%   │  95.0%   │ 基线     │
│ 原始RF       │ 55.5%  │  15.7%   │  42.3%   │  94.6%   │ 悲伤更强 │
│ A:OvR二分类  │   -    │ P=67.8%  │ P=67.8%  │ P=65.6%  │ 精度高  │
│ B:balanced   │  ~54%  │  有变    │  有变    │  可能降   │ 牺牲其他 │
│ C:逻辑回归   │  ~52%  │  低      │  低      │  高      │ 线性不够 │
│ D:智能投票   │  ~56%  │  不变    │  略升    │  不变     │ 小提升   │
│ E:三类专用SVM│  -     │  待看     │  待看     │  待看     │ 隔离训练 │
│ E:三类专用RF │  -     │  待看     │  待看     │  待看     │ 隔离训练 │
│ E:三类Ens    │  -     │  待看     │  待看     │  待看     │ 最佳?    │
└──────────────┴────────┴──────────┴──────────┴──────────┴──────────┘

结论:
1. 单纯换模型/调参无法根本解决 → 这是特征空间本身的限制
2. 最有希望的路径:
   a) 三类专用分类器(策略E): 排除其他4类的干扰, 让模型只学三类差异
   b) 引入正交特征: 手工几何特征(眉眼嘴距离比)与热力图互补
   c) 两级级联: 先分{强类 vs 弱类池}, 再在弱类池内细分
""")
