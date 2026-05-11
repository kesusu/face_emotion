"""
板端热力图 → PC训练数据格式转换器
=====================================
将 board_heatmaps_128/heatmap_cache_128_board/ 下的 9906 个独立 .npy 文件
合并为 cnn_heatmap_classifier.py 所需的 cnn_train_data_128.npy 格式

输入: 每个npy = {'heatmap': (3,128,128) float32, 'source':..., 'category':..., 'label':...}
输出: {'heatmaps': (N,3,128,128), 'labels': (N,), 'features': None, ...}

用法: python convert_board_heatmaps.py
"""
import os, sys, time, glob, json
import numpy as np

# ==================== 配置 ====================
BOARD_HEATMAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  'board_heatmaps_128', 'heatmap_cache_128_board')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          'RAF-DB', 'train', 'reports')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'cnn_train_data_128.npy')  # 覆盖旧PC数据

LABEL_ORDER = ['惊讶','恐惧','厌恶','开心','悲伤','愤怒','中性']
DIR_NAMES = ['0惊讶','1恐惧','2厌恶','3快乐','4悲伤','5愤怒','6中性']
DIR_TO_LABEL = {d: i for i, d in enumerate(DIR_NAMES)}  # '0惊讶'→0, '1恐惧'→1, ...

def main():
    print("=" * 60)
    print("板端热力图 → PC训练数据 转换")
    print("=" * 60)
    
    t0 = time.perf_counter()
    
    # 1. 扫描所有npy文件
    all_npy_files = sorted(glob.glob(os.path.join(BOARD_HEATMAP_DIR, '*.npy')))
    # 排除 _ 开头的元数据文件
    all_npy_files = [f for f in all_npy_files if not os.path.basename(f).startswith('_')]
    
    total = len(all_npy_files)
    print(f"\n扫描到 {total} 个热力图文件")
    print(f"源目录: {BOARD_HEATMAP_DIR}")
    print(f"输出:   {OUTPUT_FILE}")
    
    if total == 0:
        print("[ERROR] 未找到任何npy文件!")
        return
    
    # 2. 逐个加载 + 分类
    X_list = []
    y_list = []
    path_list = []
    per_class = {d: 0 for d in DIR_NAMES}
    fail_count = 0
    
    for idx, npy_path in enumerate(all_npy_files):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  加载中: {idx+1}/{total} ({(idx+1)/total*100:.0f}%)...", flush=True)
        
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            heatmap = data['heatmap']  # (3, 128, 128) float32
            
            # 优先用npy内部的category字段 (可靠, 不受文件名编码影响)
            category = data.get('category', '')
            if not category or category not in DIR_TO_LABEL:
                # 回退: 从文件名解析类别
                basename = os.path.basename(npy_path)
                name_without_ext = basename.rsplit('.npy', 1)[0]
                category = None
                for d in DIR_NAMES:
                    if name_without_ext.startswith(d):
                        category = d
                        break
                
                if category is None:
                    fail_count += 1
                    if fail_count <= 3:
                        print(f"  [WARN] 无法识别类别: {basename}")
                    continue
            
            label_idx = DIR_TO_LABEL[category]
            
            X_list.append(heatmap)
            y_list.append(label_idx)
            path_list.append(data.get('original_image', npy_path))
            per_class[category] += 1
            
        except Exception as e:
            fail_count += 1
            if fail_count <= 3:
                print(f"  [WARN] 加载失败 {npy_path}: {e}")
    
    # 3. 合并
    X = np.stack(X_list)   # (N, 3, 128, 128)
    y = np.array(y_list, dtype=np.int64)
    
    elapsed = time.perf_counter() - t0
    
    print(f"\n{'─'*50}")
    print(f"转换完成! 耗时 {elapsed:.1f}s")
    print(f"  成功: {len(X_list)} 张")
    print(f"  失败: {fail_count} 张")
    print(f"  形状: {X.shape}")
    print(f"\n  类别分布:")
    for d in DIR_NAMES:
        cnt = per_class[d]
        pct = cnt / len(X_list) * 100
        label = LABEL_ORDER[DIR_TO_LABEL[d]]
        print(f"    {d}({label}): {cnt} 张 ({pct:.1f}%)")
    
    # 4. 构建输出字典 (兼容 FaceEmotionHeatmapDataset 格式)
    result = {
        'heatmaps': X,                          # (N, 3, 128, 128) float32
        'labels': y,                            # (N,) int64
        'features': None,                       # 板端数据无14维特征
        'file_paths': path_list,                # 原始路径列表
        'label_order': LABEL_ORDER,
        'categories': {d: LABEL_ORDER[DIR_TO_LABEL[d]] for d in DIR_NAMES},
        'total_samples': len(X_list),
        'heatmap_size': 128,
        'source': 'board_sdk_fiboaisdk',         # 标记来源
    }
    
    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, result)
    
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n已保存: {OUTPUT_FILE}")
    print(f"  大小: {size_mb:.1f} MB")
    
    # 6. 验证: 重载确认格式正确
    print(f"\n验证: 重新加载...")
    check = np.load(OUTPUT_FILE, allow_pickle=True).item()
    assert check['heatmaps'].shape == (len(X_list), 3, 128, 128), f"形状不匹配: {check['heatmaps'].shape}"
    assert check['labels'].shape == (len(X_list),), f"标签形状不匹配"
    assert check['source'] == 'board_sdk_fiboaisdk'
    print(f"  ✅ 格式验证通过: heatmaps={check['heatmaps'].shape}, labels={check['labels'].shape}")
    
    # 7. 输出摘要JSON
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'board_sdk_fiboaisdk',
        'total_samples': len(X_list),
        'failed_samples': fail_count,
        'shape': list(X.shape),
        'dtype': str(X.dtype),
        'output_file': OUTPUT_FILE,
        'per_class': {d: {'count': per_class[d], 'label': LABEL_ORDER[DIR_TO_LABEL[d]]} for d in DIR_NAMES},
    }
    summary_path = os.path.join(OUTPUT_DIR, 'board_data_conversion_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  摘要: {summary_path}")
    
    print(f"\n✅ 全部完成! 可直接运行训练:")
    print(f"   python cnn_heatmap_classifier.py --model-type deep --epochs 200 --batch-size 64 --lr 1e-3")


if __name__ == '__main__':
    main()
