#!/usr/bin/env python3
import os
import h5py
import numpy as np

print("DEBUG: Script starting; cwd =", os.getcwd())
print("DEBUG: Contents before mkdir:", os.listdir())

# 1. Ensure output folder exists
os.makedirs("data", exist_ok=True)

# 2. Dummy parameters
N = 1000  # number of CSI frames
F = 90  # features per frame
print(f"DEBUG: N={N}, F={F}")

# 3. Generate synthetic float32 CSI data
raw = (np.random.randn(N, F) * 10).astype("float32")

# 4. Write HDF5 with fixed-shape datasets
out_path = "data/dummy_run.h5"
with h5py.File(out_path, "w") as f:
    f.create_dataset("csi_ts", data=np.linspace(0, 10, N))
    f.create_dataset("csi", data=raw, dtype="f4")
print(f"✅ Created file {out_path}")
