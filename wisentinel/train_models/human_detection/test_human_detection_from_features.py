import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

FEATURES = [
    'tof_mean', 'tof_std', 'corr_eigen_ratio',
    'num_peaks', 'std_amp', 'median_amp', 'skew_amp', 'kurtosis_amp'
]
WINDOW_SIZE = 30

def load_test_data(feature_data_dir, split, window_size=WINDOW_SIZE):
    """Load test data in windows of 30 packets each."""
    X = []
    y = []
    file_labels = []

    label_map = {
        'no_person': 0,
        'standing': 1,
        'crouching': 1
    }

    for folder, label in label_map.items():
        data_dir = os.path.join(feature_data_dir, folder, split)
        if not os.path.exists(data_dir):
            continue
        print(f"[INFO] Loading {folder} data from {data_dir}")
        for fname in os.listdir(data_dir):
            if not fname.endswith('.csv'):
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
    y = np.array(y).astype(np.float32)
    return X, y, file_labels

def main():
    base_dir = os.path.abspath("../../feature_data_2")
    model_path = "human_detection_cnn_model.h5"

    print("[INFO] Loading test data...")
    X_test, y_test, label_list = load_test_data(base_dir, 'test')
    if X_test is None:
        print("[ERROR] No test data found.")
        return

    print(f"[INFO] Loaded {X_test.shape[0]} test windows.")

    print("[INFO] Loading model...")
    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("[INFO] Predicting...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Accuracy per class
    label_arr = np.array(label_list)
    acc_total = np.mean(y_pred == y_test)
    for cls in ['no_person', 'standing', 'crouching']:
        mask = label_arr == cls
        if np.any(mask):
            acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"[RESULT] {cls} accuracy: {acc:.4f}")

    print(f"[RESULT] Overall accuracy: {acc_total:.4f}")

    print("\nSample predictions (first 10):")
    for i in range(min(10, len(y_pred))):
        print(f"True: {int(y_test[i])}, Pred: {int(y_pred[i])}, Prob: {y_pred_prob[i][0]:.3f}, Source: {label_list[i]}")

if __name__ == "__main__":
    main()
