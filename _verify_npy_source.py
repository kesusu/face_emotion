"""Verify npy source metadata"""
import numpy as np

p = r'c:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\new_train\reports\cnn_train_data_128.npy'
raw = np.load(p, allow_pickle=True).item()

print('=== ALL KEYS ===')
for k, v in raw.items():
    if isinstance(v, str):
        print(f'  [{k}] str: {v[:300]}')
    elif hasattr(v, 'shape'):
        print(f'  [{k}] array: shape={v.shape} dtype={v.dtype}')
    else:
        print(f'  [{k}] {type(v)} = {repr(v)[:200]}')

print()

# Check for source info in various possible keys
for src_key in ['source', 'meta', 'info', 'description', 'origin']:
    if src_key in raw:
        val = raw[src_key]
        if isinstance(val, dict):
            for k2, v2 in val.items():
                sv = str(v2)[:200] if not hasattr(v2, 'shape') else f'array{v2.shape}'
                print(f'  >>> {src_key}.{k2} = {sv}')
        else:
            print(f'  >>> {src_key} = {val}')

# Also check heatmap value distribution to identify PC MediaPipe pattern
hm = raw['heatmaps']
print(f'\n=== HEATMAP STATS (first 3 samples) ===')
for i in range(3):
    h = hm[i]
    print(f'  sample[{i}]: min={h.min():.4f} max={h.max():.4f} mean={h.mean():.6f} std={h.std():.6f}')

print(f'\n  ALL: min={hm.min():.4f} max={hm.max():.4f} mean={hm.mean():.6f}')
