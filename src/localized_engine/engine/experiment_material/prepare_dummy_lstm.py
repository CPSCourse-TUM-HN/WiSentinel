import h5py, numpy as np

# Load raw CSI frames
with h5py.File("data/dummy_run.h5", "r") as f:
    raw_bytes = f["csi"][:]
N = len(raw_bytes)
# Convert each vlen-bytes entry back into a float32 vector
F = len(raw_bytes[0]) // 4
raw = np.vstack([np.frombuffer(raw_bytes[i], dtype="float32") for i in range(N)])

# Sliding windows
W, S = 50, 10
windows, labels = [], []
for start in range(0, N - W, S):
    w = raw[start : start + W, :]  # shape (W, F)
    windows.append(w)
    labels.append(np.linalg.norm(w.mean(axis=0)))  # dummy “distance”

X = np.stack(windows)  # (M, W, F)
y = np.array(labels)  # (M,)

# Split 80/20
m = len(y)
split = int(0.8 * m)
np.save("X_train.npy", X[:split])
np.save("y_train.npy", y[:split])
np.save("X_val.npy", X[split:])
np.save("y_val.npy", y[split:])
print("Saved X_train/y_train and X_val/y_val")
