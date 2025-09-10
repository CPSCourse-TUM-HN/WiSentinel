import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Feature list
FEATURES = [
    'tof_mean', 'tof_std', 'corr_eigen_ratio',
    'num_peaks', 'std_amp', 'median_amp', 'skew_amp', 'kurtosis_amp'
]

WINDOW_SIZE = 30

def load_pose_data(feature_data_dir, split, window_size=WINDOW_SIZE):
    """Loads only crouching and standing, builds windows, returns X and integer labels."""
    X = []
    y = []

    label_map = {
        "standing": 0,
        "crouching": 1
    }

    for label_name, label_val in label_map.items():
        class_dir = os.path.join(feature_data_dir, label_name, split)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(class_dir, fname))
            df = df.dropna(subset=FEATURES)
            feature_array = df[FEATURES].values
            if len(feature_array) < window_size:
                continue
            for start in range(len(feature_array) - window_size + 1):
                window = feature_array[start:start + window_size]
                X.append(window)
                y.append(label_val)

    if not X:
        return None, None

    X = np.stack(X).astype(np.float32)
    y = np.array(y).astype(np.int32)
    return X, y

def build_pose_cnn(input_shape):
    """Builds a 2-class CNN for pose estimation."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')  # Two output classes: standing, crouching
    ])
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    base_dir = os.path.abspath("../../feature_data_2")
    print(f"[INFO] Using feature data directory: {base_dir}")

    print("[INFO] Loading training data...")
    X_train, y_train = load_pose_data(base_dir, 'train')
    print("[INFO] Loading validation data...")
    X_val, y_val = load_pose_data(base_dir, 'validation')
    print("[INFO] Loading test data...")
    X_test, y_test = load_pose_data(base_dir, 'test')

    if X_train is None or X_val is None or X_test is None:
        print("[ERROR] Not enough data to train pose estimation model.")
        return

    print(f"[INFO] Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    model = build_pose_cnn((WINDOW_SIZE, len(FEATURES)))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("[INFO] Training pose estimation CNN...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    print("[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Pose Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    model.save("pose_estimation_cnn_model.h5")
    print("[INFO] Model saved as pose_estimation_cnn_model.h5")

if __name__ == "__main__":
    main()
