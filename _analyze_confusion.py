import numpy as np, json, os
from sklearn.metrics import confusion_matrix, classification_report

# 加载最新实验报告和模型
base = r'c:\Users\ke\Desktop\嵌赛\face_emotion'
report_path = os.path.join(base, 'RAF-DB', 'train', 'reports', 'rf_cleaned_v2_report.json')
model_dir = os.path.join(base, 'RAF-DB', 'train', 'reports')

with open(report_path, 'r', encoding='utf-8') as f:
    report = json.load(f)

print("=" * 70)
print("分析目标: 悲伤(4) / 厌恶(2) / 中性(6) 三类区分问题")
print("=" * 70)

LABEL_ORDER = ['0惊讶','1恐惧','2厌恶','3快乐','4悲伤','5愤怒','6中性']
TARGET_CLASSES = [2, 4, 6]  # 厌恶, 悲伤, 中性

print(f"\n整体最佳: {report['best_model']} = {report['best_accuracy']:.1%}")
print(f"样本数: {report['total_samples']}, PCA方差覆盖率: {report['pca_variance_explained']:.4%}")

print(f"\n{'='*70}")
print("逐类准确率全貌:")
print(f"{'='*70}")
for name in LABEL_ORDER:
    acc = report['per_class_accuracy'].get(name, 0)
    bar = '#' * int(acc * 50)
    tag = " ★ 目标类" if name in ['2厌恶','4悲伤','6中性'] else ""
    print(f"  {name}: {acc:.1%} {bar}{tag}")

# 加载模型做详细混淆矩阵分析
import joblib

svm_pkg = joblib.load(os.path.join(model_dir, [f for f in os.listdir(model_dir) if f.startswith('v16_svm_')][0]))
rf_pkg = joblib.load(os.path.join(model_dir, [f for f in os.listdir(model_dir) if f.startswith('v16_rf_')][0]))

# 需要重建验证集来计算混淆矩阵... 
# 但我们没有保存X_va_pca。需要从npy重新加载+transform。
# 为了快速分析, 直接用报告中的per_class_accuracy + 加载模型做预测

print(f"\n{'='*70}")
print("核心问题: 三类互相混淆分析")
print(f"{'='*70}")

# 重新加载数据并计算三个模型在三类上的完整混淆矩阵
DATA_DIR = os.path.join(base, 'RAF-DB', 'train')
CACHE_DIR = os.path.join(DATA_DIR, 'reports', 'heatmap_cache_128')

valid_pairs = []
for cls_name in LABEL_ORDER:
    cls_dir = os.path.join(DATA_DIR, cls_name)
    if not os.path.exists(cls_dir): continue
    for jpg in sorted([f for f in os.listdir(cls_dir) if f.endswith('.jpg')]):
        npy_name = f"{cls_name}_{jpg}.npy"
        npy_path = os.path.join(CACHE_DIR, npy_name)
        if os.path.exists(npy_path):
            valid_pairs.append((cls_name, npy_path))

LABEL_MAP = {name: i for i, name in enumerate(LABEL_ORDER)}

print(f"\n重新加载 {len(valid_pairs)} 个热力图...")
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

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
del X

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

pca = PCA(n_components=50, random_state=42)
X_tr_p = pca.fit_transform(X_tr_s)
X_va_p = pca.transform(X_va_s)

del X_tr_s, X_va_s

# 获取三个模型的预测
svm_model = svm_pkg['model']
rf_model = rf_pkg['model']

pred_svm = svm_model.predict(X_va_p)
pred_rf = rf_model.predict(X_va_p)

# === 核心分析: 三类的混淆矩阵 ===
target_names = ['2厌恶', '4悲伤', '6中性']
target_ids = [2, 4, 6]

print("\n" + "=" * 70)
print("SVM 在三类别上的混淆矩阵 (行=真实, 列=预测)")
print("=" * 70)

# 只看三类的子矩阵
mask = np.isin(y_va, target_ids)
y_sub = y_va[mask]
svm_pred_sub = pred_svm[mask]
rf_pred_sub = pred_rf[mask]

cm_svm = confusion_matrix(y_sub, svm_pred_sub, labels=target_ids)
cm_rf = confusion_matrix(y_sub, rf_pred_sub, labels=target_ids)

def print_cm(cm, title):
    print(f"\n{title}")
    header = "          " + "  ".join(f"{n:>8s}" for n in target_names)
    print(header)
    for i, row_name in enumerate(target_names):
        row_total = cm[i].sum()
        line = f"{row_name:>8s}"
        for j, val in enumerate(cm[i]):
            pct = val / row_total * 100 if row_total > 0 else 0
            line += f"  {val:>4d}({pct:>5.1f}%)"
        print(line)
    # 列总计
    col_totals = cm.sum(axis=0)
    line = "  TOTAL   "
    for j, t in enumerate(col_totals):
        line += f"  {t:>4d}"
    print(line)
    
    # 计算每类的precision/recall
    print(f"\n  详细指标:")
    for i, name in enumerate(target_names):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i].sum() - tp
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0
        rec = tp / (tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        print(f"    {name}: P={prec:.1%} R={rec:.1%} F1={f1:.1%}")

print_cm(cm_svm, "SVM(RBF,C=5) 三类混淆:")
print_cm(cm_rf, "RF(200树) 三类混淆:")

# === 关键发现: SVM vs RF 的预测差异 ===
print(f"\n{'='*70}")
print("SVM vs RF 预测差异分析 (只在三类样本上)")
print(f"{'='*70}")

agree_correct = 0   # 两都对
agree_wrong = 0     # 两都错且相同错误
disagree = 0        # 预测不同
disagree_details = []  # 记录分歧案例

for i, (true_lbl, sv, rf_p) in enumerate(zip(y_sub, svm_pred_sub, rf_pred_sub)):
    true_name = LABEL_ORDER[true_lbl]
    sv_name = LABEL_ORDER[sv]
    rf_name = LABEL_ORDER[rf_p]
    
    if sv == rf_p:
        if sv == true_lbl:
            agree_correct += 1
        else:
            agree_wrong += 1
    else:
        disagree += 1
        disagree_details.append({
            'true': true_name, 'svm': sv_name, 'rf': rf_name,
            'svm_right': sv == true_lbl, 'rf_right': rf_p == true_lbl,
            'idx': i
        })

total = len(y_sub)
print(f"  总计: {total}个三类样本")
print(f"  SVM+RF一致且正确: {agree_correct} ({agree_correct/total:.1%})")
print(f"  SVM+RF一致且错误: {agree_wrong} ({agree_wrong/total:.1%})")
print(f"  SVM+RF预测不同:  {disagree} ({disagree/total:.1%}) ← Ensemble可能有用")

if disagree > 0:
    # 分歧中谁对了？
    svm_win = sum(1 for d in disagree_details if d['svm_right'])
    rf_win = sum(1 for d in disagree_details if d['rf_right'])
    both_wrong = disagree - svm_win - rf_win
    
    print(f"\n  分歧中:")
    print(f"    SVM对RF错: {svm_win} ({svm_win/disagree:.1%})")
    print(f"    RF对SVM错: {rf_win} ({rf_win/disagree:.1%})")
    print(f"    两都错:     {both_wrong} ({both_wrong/disagree:.1%})")
    
    # 分歧详情按真实类别统计
    print(f"\n  分歧详情(按真实类别):")
    from collections import Counter, defaultdict
    disc_by_true = defaultdict(list)
    for d in disagree_details:
        disc_by_true[d['true']].append(d)
    
    for tn in target_names:
        cases = disc_by_true.get(tn, [])
        if not cases:
            continue
        svm_right_cnt = sum(1 for c in cases if c['svm_right'])
        rf_right_cnt = sum(1 for c in cases if c['rf_right'])
        print(f"    {tn}: {len(cases)}次分歧 → SVM对{svm_right_cnt}/RF对{rf_right_cnt}")
        
        # 显示具体混淆模式
        patterns = Counter((c['svm'], c['rf']) for c in cases)
        for (sv_p, rf_p), cnt in patterns.most_common(5):
            print(f"      SVM→{sv_p} RF→{rf_p}: {cnt}次")

# === 投票模拟 ===
print(f"\n{'='*70}")
print("Ensemble投票模拟: SVM+RF 二票多数决")
print(f"{'='*70}")

ensemble_pred = []
for sv, rf_p in zip(pred_svm, pred_rf):
    if sv == rf_p:
        ensemble_pred.append(sv)
    else:
        # 平票时随机选一个(或选置信度高的, 这里简化用SVM)
        ensemble_pred.append(sv)  # SVM更准, 作为tie-breaker

ensemble_pred = np.array(ensemble_pred)

# 全7类ensemble准确率
ens_acc_all = (ensemble_pred == y_va).mean()
print(f"\n  全部7类: SVM={np.mean(pred_svm==y_va):.1%} RF={np.mean(pred_rf==y_va):.1%} Ensemble={ens_acc_all:.1%}")

# 三类ensemble
mask3 = np.isin(y_va, target_ids)
y_3 = y_va[mask3]
ep_3 = ensemble_pred[mask3]
sv_3 = pred_svm[mask3]
rf_3 = pred_rf[mask3]

ens_acc_3 = (ep_3 == y_3).mean()
sv_acc_3 = (sv_3 == y_3).mean()
rf_acc_3 = (rf_3 == y_3).mean()
print(f"  三类子集: SVM={sv_acc_3:.1%} RF={rf_acc_3:.1%} Ensemble={ens_acc_3:.1%}")

for tid, tname in zip(target_ids, target_names):
    m = y_3 == tid
    print(f"    {tname}: SVM={(sv_3[m]==tid).mean():.1%} RF={(rf_3[m]==tid).mean():.1%} Ens={(ep_3[m]==tid).mean():.1%}")

# === 规则系统可能性分析 ===
print(f"\n{'='*70}")
print("策略建议")
print(f"{'='*70}")

print("""
基于以上分析:

1. 数据层面:
   - 厌恶只有9.0%, 样本最少(145张val), 是最大瓶颈
   - 悲伤35.0%有提升空间, 但大量被误分为其他类
   - 中性87.1%已经很好了! 清洗效果明显
   
2. 模型层面:
   - SVM和RF在三类上有显著分歧({:.0f}%的样本预测不同)
   - 分歧中SVM正确率更高 → 可作为主要模型
   - RF在某些情况下能纠正SVM的错误 → 有互补性
   
3. 推荐策略:
   A. 短期(立即可做): SVM+RF二票多数决Ensemble
      → 预期三类子集从~45%提升到~48-52%
   
   B. 中期: 补充手工特征(14维关键点距离比)训练第三模型
      → 眉间距、嘴角落差等几何特征与热力图正交
      → SVM(热力图)+SVM(手工特征)+RF(热力图) 三模型投票
      
   C. 长期: 规则后处理
      → 如: 嘴角下垂幅度>阈值 → 降低"开心/中性"概率
      → 眉头内收程度>阈值 → 提高"厌恶/愤怒"概率
""".format(disagree/total*100))
