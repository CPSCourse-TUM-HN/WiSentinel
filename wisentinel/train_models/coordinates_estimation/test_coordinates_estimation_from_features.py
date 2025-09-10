import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Updated features (without variance features)
FEATURES = [
    'tof_mean', 'tof_std', 'aoa_peak',
    'corr_eigen_ratio', 'std_amp',
    'median_amp', 'skew_amp'
]

WINDOW_SIZE = 30

positions_to_coordinates = {
    "pos1": (0, 2), "pos2": (1, 2), "pos3": (2, 2),
    "pos4": (2, 1), "pos5": (2, 0), "pos6": (1, 0),
    "pos7": (0, 0), "pos8": (0, 1), "posX": (1, 1)
}

def extract_pos_from_filename(filename):
    match = re.search(r'pos[1-8X]', filename)
    return match.group(0) if match else None

def load_test_data(split_dirs, window_size=WINDOW_SIZE):
    """Loads windowed test samples and true coordinates."""
    X = []
    y = []
    sources = []

    for folder in split_dirs:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith(".csv"):
                continue
            pos = extract_pos_from_filename(fname)
            if pos not in positions_to_coordinates:
                continue
            coords = positions_to_coordinates[pos]
            df = pd.read_csv(os.path.join(folder, fname))
            df = df.dropna(subset=FEATURES)
            feature_array = df[FEATURES].values
            if len(feature_array) < window_size:
                continue
            for start in range(len(feature_array) - window_size + 1):
                window = feature_array[start:start + window_size]
                X.append(window)
                y.append(coords)
                sources.append(f"{fname}")

    if not X:
        return None, None, None

    return np.stack(X).astype(np.float32), np.array(y).astype(np.float32), sources

def main():
    base_dir = os.path.abspath("../../feature_data_2")
    test_dirs = [
        os.path.join(base_dir, "standing", "test"),
        os.path.join(base_dir, "crouching", "test")
    ]
    model_path = "coordinates_cnn_model.h5"

    print("[INFO] Loading test data...")
    X_test, y_test, sources = load_test_data(test_dirs)
    if X_test is None:
        print("[ERROR] No test data found.")
        return

    print(f"[INFO] Loaded {X_test.shape[0]} test samples")

    print("[INFO] Loading trained model...")
    model = load_model(model_path)

    print("[INFO] Making predictions...")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[RESULT] Test MSE: {mse:.4f}")
    print(f"[RESULT] Test MAE: {mae:.4f}")

    print("\n[INFO] Sample predictions:")
    for i in range(min(10, len(y_test))):
        true_x, true_y = y_test[i]
        pred_x, pred_y = y_pred[i]
        print(f"{sources[i]:40s} | True: ({true_x:.2f}, {true_y:.2f}) | Pred: ({pred_x:.2f}, {pred_y:.2f})")

if __name__ == "__main__":
    main()
