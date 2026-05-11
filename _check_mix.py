"""检查板子上0惊讶目录是否有不该存在的文件（来自其他类别）"""
import os, subprocess

PC_BASE = r"C:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train"
DEVICE = "28de40d2"
BOARD_BASE = "/home/fibo/AI model/cv_models/pose/photos/train"

# 1. 拉取板子0惊讶完整列表
result = subprocess.run(
    ["adb", "-s", DEVICE, "shell", f"ls '{BOARD_BASE}/0惊讶'"],
    capture_output=True, text=True
)
board_files = set()
for line in result.stdout.split('\n'):
    name = line.strip()
    if name.endswith('.jpg') or name.endswith('.png'):
        board_files.add(name)

# 2. PC端0惊讶所有文件
pc_dir = os.path.join(PC_BASE, "0惊讶")
pc_files = set(f for f in os.listdir(pc_dir) if f.endswith(('.jpg', '.png')))

# 3. PC端其他类别的所有文件（用于检查是否混入）
other_cat_files = {}
all_other = set()
for cat in ["1恐惧", "2厌恶", "3快乐", "4悲伤", "5愤怒", "6中性"]:
    cdir = os.path.join(PC_BASE, cat)
    files = set(f for f in os.listdir(cdir) if f.endswith(('.jpg', '.png')))
    other_cat_files[cat] = files
    all_other |= files

print(f"=== 交叉检查 ===")
print(f"板子 0惊讶: {len(board_files)} 张")
print(f"PC端 0惊讶: {len(pc_files)} 张")

# 板子有但PC没有的（可疑！）
extra_on_board = board_files - pc_files
if extra_on_board:
    print(f"\n!!! 警告: 板子上有 {len(extra_on_board)} 张PC端0惊讶里没有的文件 !!!")
    # 检查这些是否来自其他类别
    for f in sorted(extra_on_board):
        found_in = []
        for cat, files in other_cat_files.items():
            if f in files:
                found_in.append(cat)
        if found_in:
            print(f"  [错误!] {f} -> 属于类别: {found_in}")
        else:
            print(f"  [未知来源] {f}")
else:
    print("\n✅ 板子上的文件全部在PC端0惊讶中找到，没有外来文件")

# PC有但板上没有的
missing_on_board = pc_files - board_files
if missing_on_board:
    print(f"\nPC有但板子缺少: {len(missing_on_board)} 张")
else:
    print(f"\n✅ PC端文件在板上齐全")
