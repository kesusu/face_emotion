import os
cache = r'c:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\reports\heatmap_cache_128'

root_items = sorted(os.listdir(cache))
subdirs = [d for d in root_items if os.path.isdir(os.path.join(cache, d))]
files = [f for f in root_items if os.path.isfile(os.path.join(cache, f))]

print(f'Root items: {len(root_items)}')
print(f'Subdirs ({len(subdirs)}):', subdirs[:10])
print(f'Files at root ({len(files)})')
for f in sorted(files)[:5]:
    print(f'  {f}')

# Check first subdir contents
if subdirs:
    sd = os.path.join(cache, subdirs[0])
    items = os.listdir(sd)
    print(f'\nSubdir "{subdirs[0]}" has {len(items)} items:')
    print('  Sample:', sorted(items)[:5])

# Check cnn_train_data_128
npy_path = r'c:\Users\ke\Desktop\嵌赛\face_emotion\RAF-DB\train\reports\cnn_train_data_128.npy'
if os.path.exists(npy_path):
    sz = os.path.getsize(npy_path) // 1024 // 1024
    print(f'\ncnn_train_data_128.npy exists: {sz}MB')
else:
    print('\ncnn_train_data_128.npy NOT FOUND')
