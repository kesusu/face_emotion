"""
同步板端筛选后的数据: 对比PC端RAF-DB与板端photos/train的文件名
删除PC端多出的(被板端删掉的)脏数据
"""

import os

BASE_PC = r'c:\Users\ke\Desktop\嵌赛\face_emotion'

# 类别: (train子目录, valid子目录, 板端目录, 通用名)
CATEGORIES = [
    ('0', '0惊讶', '0惊讶', '惊讶'),
    ('1', '1恐惧', '1恐惧', '恐惧'),
    ('2', '2厌恶', '2厌恶', '厌恶'),
    ('3', '3快乐', '3快乐', '开心'),
    ('4', '4悲伤', '4悲伤', '悲伤'),
    ('5', '5愤怒', '5愤怒', '愤怒'),
    ('6', '6中性', '6中性', '中性'),
]

total_kept = 0
total_deleted = 0
summary = []

for train_sub, valid_sub, board_dir, name in CATEGORIES:
    # === 读取板端文件名白名单 ===
    board_file = os.path.join(BASE_PC, f'tmp_board_{train_sub}.txt')
    if not os.path.exists(board_file):
        print(f"[跳过] {name}: 板端文件列表不存在")
        continue
    
    with open(board_file, 'r', encoding='utf-8') as f:
        board_files = set(line.strip() for line in f if line.strip())
    
    # === 处理 train 目录 ===
    pc_train_dir = os.path.join(BASE_PC, 'RAF-DB', 'train', train_sub)
    if os.path.exists(pc_train_dir):
        pc_files = set(f for f in os.listdir(pc_train_dir) if f.endswith('.jpg'))
        to_delete = pc_files - board_files
        kept = pc_files & board_files
        
        d_cnt, k_cnt = len(to_delete), len(kept)
        total_deleted += d_cnt
        total_kept += k_cnt
        summary.append(f"  train/{name}: {len(pc_files)}张 -> 保留{k_cnt} 删除{d_cnt}")
        
        if d_cnt > 0:
            for fname in sorted(to_delete):
                os.remove(os.path.join(pc_train_dir, fname))
            print(f"  [train/{name}] 已删除 {d_cnt} 个脏文件")
        else:
            print(f"  [train/{name}] 无需删除 ({k_cnt}张全部保留)")
    
    # === 处理 valid 目录 (同样用板端白名单过滤) ===
    pc_valid_dir = os.path.join(BASE_PC, 'RAF-DB', 'valid', valid_sub)
    if os.path.exists(pc_valid_dir):
        pc_vfiles = set(f for f in os.listdir(pc_valid_dir) if f.endswith('.jpg'))
        to_del_v = pc_vfiles - board_files
        
        if to_del_v:
            print(f"  [valid/{name}] 删除 {len(to_del_v)} 个脏文件...")
            for fname in sorted(to_del_v):
                fp = os.path.join(pc_valid_dir, fname)
                if os.path.exists(fp):
                    os.remove(fp)

print("\n" + "="*50)
print("汇总:")
for s in summary:
    print(s)
print(f"\n总计保留: {total_kept} 张 | 总计删除: {total_deleted} 张脏数据")

# 清理临时文件
for i in range(7):
    tmp = os.path.join(BASE_PC, f'tmp_board_{i}.txt')
    if os.path.exists(tmp):
        os.remove(tmp)

print("\n完成! PC端数据已与板端同步.")
