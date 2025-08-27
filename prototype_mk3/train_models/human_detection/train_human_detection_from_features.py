import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight


# Features to use for training
FEATURES = [
    'tof_mean', 'tof_std', 'corr_eigen_ratio',
    'num_peaks', 'std_amp', 'median_amp', 'skew_amp', 'kurtosis_amp'
]

WINDOW_SIZE = 30

def load_data(feature_data_dir, split, window_size=WINDOW_SIZE):
    """Loads per-packet CSVs, groups packets into windows, and returns CNN input format."""
    X = []
    y = []

    label_map = {
        "no_person": 0,
        "standing": 1,
        "crouching": 1
    }

    for label_name, label_val in label_map.items():
        class_dir = os.path.join(feature_data_dir, label_name, split)
        if not os.path.exists(class_dir):
            print(f"[WARNING] Directory {class_dir} does not exist, skipping.")
            continue

        print(f"[INFO] Loading data from {class_dir} for label '{label_name}'")
        print ("Found", len(os.listdir(class_dir)), "files")
        for fname in os.listdir(class_dir):
            if not fname.endswith('.csv'):
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

    X = np.stack(X)
    y = np.array(y)

    print("[INFO] Label distribution:")
    print("  no_person :", np.sum(y == 0))
    print("  person    :", np.sum(y == 1))

    return X, y

def build_cnn_model(input_shape):
    """Builds a simple 1D CNN for binary classification."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Output: probability of human presence
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    base_dir = os.path.abspath("../../feature_data_2")
    print(f"[INFO] Using feature data directory: {base_dir}")

    print("[INFO] Loading training data...")
    X_train, y_train = load_data(base_dir, 'train')
    print("[INFO] Loading validation data...")
    X_val, y_val = load_data(base_dir, 'validation')
    print("[INFO] Loading test data...")
    X_test, y_test = load_data(base_dir, 'test')

    if X_train is None or X_val is None or X_test is None:
        print("[ERROR] Not enough data for training/validation/testing.")
        return

    # Ensure correct dtype
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    print(f"[INFO] Training samples: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    model = build_cnn_model((WINDOW_SIZE, len(FEATURES)))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("[INFO] Training model...")
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=2
    )

    print("[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Save model
    model.save("human_detection_cnn_model.h5")
    print("[INFO] Model saved as human_detection_cnn_model.h5")

if __name__ == "__main__":
    main()
