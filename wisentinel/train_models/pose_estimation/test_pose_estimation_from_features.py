import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Feature columns used in training
FEATURES = [
    'tof_mean', 'tof_std', 'corr_eigen_ratio',
    'num_peaks', 'std_amp', 'median_amp', 'skew_amp', 'kurtosis_amp'
]

LABEL_MAP = {
    0: 'standing',
    1: 'crouching'
}

WINDOW_SIZE = 30

def load_test_data(feature_data_dir, split, window_size=WINDOW_SIZE):
    """Loads test data windows from crouching and standing samples."""
    X = []
    y = []
    file_labels = []

    class_map = {
        'standing': 0,
        'crouching': 1
    }

    for folder, label in class_map.items():
        data_dir = os.path.join(feature_data_dir, folder, split)
        if not os.path.exists(data_dir):
            continue
        print(f"[INFO] Loading {folder} from {data_dir}")
        for fname in os.listdir(data_dir):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(data_dir, fname))
            df = df.dropna(subset=FEATURES)
            feature_array = df[FEATURES].values
            if len(feature_array) < window_size:
                continue
            for start in range(len(feature_array) - window_size + 1):
                window = feature_array[start:start + window_size]
                X.append(window)
                y.append(label)
                file_labels.append(folder)

    if not X:
        return None, None, None

    X = np.stack(X).astype(np.float32)
    y = np.array(y).astype(np.int32)
    return X, y, file_labels

def main():
    base_dir = os.path.abspath("../../feature_data_2")
    model_path = "pose_estimation_cnn_model.h5"

    print("[INFO] Loading test data...")
    X_test, y_test, file_labels = load_test_data(base_dir, 'test')
    if X_test is None:
        print("[ERROR] No test data found.")
        return

    print(f"[INFO] Loaded {X_test.shape[0]} windows.")

    print("[INFO] Loading model...")
    model = load_model(model_path)

    print("[INFO] Predicting...")
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    # Accuracy
    acc_total = np.mean(y_pred == y_test)
    print(f"[RESULT] Overall accuracy: {acc_total:.4f}")

    for class_idx in [0, 1]:
        mask = (y_test == class_idx)
        if np.any(mask):
            acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"[RESULT] {LABEL_MAP[class_idx]} accuracy: {acc:.4f}")

    print("\n[INFO] Sample predictions (first 10):")
    for i in range(min(10, len(y_pred))):
        print(f"True: {LABEL_MAP[y_test[i]]:10s} | Predicted: {LABEL_MAP[y_pred[i]]:10s} | Source: {file_labels[i]}")

if __name__ == "__main__":
    main()
