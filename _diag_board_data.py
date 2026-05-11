"""诊断板子热力图数据质量"""
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          'RAF-DB', 'train', 'reports', 'cnn_train_data_128.npy')

data = np.load(DATA_PATH, allow_pickle=True).item()
X = data['heatmaps']  # (N, 3, 128, 128)
y = data['labels']

print("=" * 60)
print(f"Source: {data.get('source', '?')}")
print(f"Shape: {X.shape}, Dtype: {X.dtype}")
print(f"Total: {len(X)} samples")
print("=" * 60)

# Per-channel stats
ch_names = ['Ch0:XY位置', 'Ch1:Z深度', 'Ch2:区域分组']
for ch in range(3):
    d = X[:, ch]
    print(f"\n{ch_names[ch]}:")
    print(f"  mean={d.mean():.6f}, std={d.std():.6f}")
    print(f"  min={d.min():.4f}, max={d.max():.4f}")
    print(f"  p25={np.percentile(d,25):.4f}, median={np.median(d):.4f}, p75={np.percentile(d,75):.4f}")
    print(f"  near-zero (<0.01): {(d < 0.01).mean()*100:.1f}%")

# Per-class intensity
print("\n" + "=" * 60)
print("Per-Class Heatmap Intensity (mean across all pixels)")
print("=" * 60)
labels_cn = ['惊讶','恐惧','厌恶','开心','悲伤','愤怒','neutral']
for c in range(7):
    mask = y == c
    xc = X[mask]
    print(f"  {labels_cn[c]:>4}(N={mask.sum():>5}): mean_all={xc.mean():.4f} | "
          f"Ch0={xc[:,0].mean():.4f} Ch1={xc[:,1].mean():.4f} Ch2={xc[:,2].mean():.4f}")

# Check for anomalies: any class with very low/very high intensity?
print("\n" + "=" * 60)
print("Key Observations:")
print("=" * 60)

# Check if all heatmaps are similar-looking (low variance between samples)
sample_variance = X.var(axis=0).mean()  # variance of each pixel position across samples
global_mean = X.mean()
print(f"  Global mean pixel value: {global_mean:.4f}")
print(f"  Cross-sample variance: {sample_variance:.6f}")

# Check per-sample mean (to see if some samples are blank/dark)
per_sample_means = X.mean(axis=(1,2,3))  # (N,)
print(f"  Per-sample mean range: [{per_sample_means.min():.4f}, {per_sample_means.max():.4f}]")

# Dark samples check
dark_threshold = 0.05
dark_count = (per_sample_means < dark_threshold).sum()
print(f"  Very dark samples (mean<{dark_threshold}): {dark_count} ({dark_count/len(X)*100:.1f}%)")

# Bright samples check  
bright_count = (per_sample_means > 0.5).sum()
print(f"  Very bright samples (mean>0.5): {bright_count} ({bright_count/len(X)*100:.1f}%)")
