import numpy as np
import os

files = [
    r"RAF-DB\new_train\reports\cnn_train_data_128.npy",
    r"RAF-DB\new_train\reports\cnn_train_data_pc_mediapipe.npy",
]

for f in files:
    fp = os.path.join(os.path.dirname(__file__), f)
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True).item()
        n = len(d["images"])
        nl = len(set(d["labels"]))
        sz_mb = os.path.getsize(fp) / 1024 / 1024
        print(f"{f}: {n} samples, {nl} classes, {sz_mb:.0f}MB")
    else:
        print(f"{f}: NOT FOUND")
