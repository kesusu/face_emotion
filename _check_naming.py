import os
base = r'c:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train'
cache = os.path.join(base, 'reports', 'heatmap_cache_128')

for cls in ['0惊讶','1恐惧']:
    npy_dir = os.path.join(cache, cls)
    jpg_dir = os.path.join(base, cls)
    
    npys = sorted(os.listdir(npy_dir))[:5] if os.path.exists(npy_dir) else []
    jpgs = sorted([f for f in os.listdir(jpg_dir) if f.endswith('.jpg')])[:5] if os.path.exists(jpg_dir) else []
    
    print(f'=== {cls} ===')
    print(f'npy samples: {npys}')
    print(f'jpg samples: {jpgs}')
    print(f'total npy in dir: {len(os.listdir(npy_dir)) if os.path.exists(npy_dir) else 0}')
    print(f'total jpg in dir: {len([f for f in os.listdir(jpg_dir) if f.endswith(".jpg")]) if os.path.exists(jpg_dir) else 0}')
