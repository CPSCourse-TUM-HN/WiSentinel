import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Mapping positions to physical (x, y) coordinates in meters
positions_to_coordinates = {
    "pos1": (0, 2), "pos2": (1, 2), "pos3": (2, 2),
    "pos4": (2, 1), "pos5": (2, 0), "pos6": (1, 0),
    "pos7": (0, 0), "pos8": (0, 1), "posX": (1, 1)
}

WINDOW_SIZE = 30

# Features used for coordinates model
FEATURES = ['tof_mean', 'tof_std', 'aoa_peak', 'corr_eigen_ratio', 'std_amp', 'median_amp', 'skew_amp']

def extract_pos_from_filename(filename):
    """Extracts position label like pos1, posX, etc."""
    match = re.search(r'pos[1-8X]', filename)
    return match.group(0) if match else None

def load_windowed_data(split_folder, window_size=WINDOW_SIZE):
    """Loads data from split folder as (N_windows, window_size, feature_count), (N_windows, 2)"""
    X, y = [], []
    for fname in os.listdir(split_folder):
        if not fname.endswith('.csv'):
            continue
        pos = extract_pos_from_filename(fname)
        if pos not in positions_to_coordinates:
            continue
        coords = positions_to_coordinates[pos]
        df = pd.read_csv(os.path.join(split_folder, fname))
        df = df.dropna(subset=FEATURES)
        feat_array = df[FEATURES].values
        if len(feat_array) < window_size:
            continue
        for start in range(len(feat_array) - window_size + 1):
            window = feat_array[start:start + window_size]
            X.append(window)
            y.append(coords)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y).astype(np.float32)

def build_coordinates_cnn(input_shape):
    """Builds CNN model to predict (x, y) from CSI feature windows."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='linear')  # Output: (x, y)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def main():
    base_path = os.path.abspath("../../feature_data_2")
    train_dirs = [os.path.join(base_path, "standing", "train"), os.path.join(base_path, "crouching", "train")]
    val_dirs = [os.path.join(base_path, "standing", "validation"), os.path.join(base_path, "crouching", "validation")]
    test_dirs = [os.path.join(base_path, "standing", "test"), os.path.join(base_path, "crouching", "test")]

    print("[INFO] Loading training data...")
    X_train, y_train = zip(*[load_windowed_data(d) for d in train_dirs if os.path.exists(d)])
    X_train = np.vstack([x for x in X_train if x is not None])
    y_train = np.vstack([y for y in y_train if y is not None])

    print("[INFO] Loading validation data...")
    X_val, y_val = zip(*[load_windowed_data(d) for d in val_dirs if os.path.exists(d)])
    X_val = np.vstack([x for x in X_val if x is not None])
    y_val = np.vstack([y for y in y_val if y is not None])

    print("[INFO] Loading test data...")
    X_test, y_test = zip(*[load_windowed_data(d) for d in test_dirs if os.path.exists(d)])
    X_test = np.vstack([x for x in X_test if x is not None])
    y_test = np.vstack([y for y in y_test if y is not None])

    print(f"[INFO] Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    model = build_coordinates_cnn((WINDOW_SIZE, len(FEATURES)))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("[INFO] Training CNN...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    print("[INFO] Evaluating on test set...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test MSE: {loss:.4f}, MAE: {mae:.4f}")

    model.save("coordinates_cnn_model.h5")
    print("[INFO] Model saved as coordinates_cnn_model.h5")

if __name__ == "__main__":
    main()
