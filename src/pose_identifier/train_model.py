import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- CONFIGURATION ---
# We will standardize all spectrograms to have this many time steps.
# From the error, we know some are longer than the STFT window (100).
# A value around 30-40 is typical for these short gestures.
FIXED_TIME_STEPS = 40


def load_data(data_directory: str):
    """
    Loads all .npy files, standardizing them to a fixed length.
    """
    samples = []
    labels = []

    print(f"Loading and standardizing data from '{data_directory}'...")

    filenames = os.listdir(data_directory)
    for filename in filenames:
        if filename.endswith("_data.npy"):
            prefix = filename.replace("_data.npy", "")
            data_path = os.path.join(data_directory, filename)

            # Load the spectrogram data
            data = np.load(data_path)

            # --- THE FIX: Standardize the time dimension (width) of the spectrogram ---
            current_time_steps = data.shape[1]

            if current_time_steps > FIXED_TIME_STEPS:
                # Truncate the data if it's too long
                data = data[:, :FIXED_TIME_STEPS, :]
            elif current_time_steps < FIXED_TIME_STEPS:
                # Pad with zeros if it's too short
                pad_width = FIXED_TIME_STEPS - current_time_steps
                # The padding is applied to the time dimension (axis 1)
                data = np.pad(data, ((0, 0), (0, pad_width), (0, 0)), mode="constant")

            # Now all 'data' arrays will have the shape (freq_bins, FIXED_TIME_STEPS, channels)
            samples.append(data)

            # Extract and append the label
            try:
                label = int(prefix.split("-")[1])
                labels.append(label - 1)
            except (ValueError, IndexError) as e:
                print(f"Could not parse label from filename: {filename}. Error: {e}")
                samples.pop()

    print(f"Loaded and standardized {len(samples)} samples.")
    return np.array(samples), np.array(labels)


def build_model(input_shape, num_classes: int):
    # ... (no changes to this function)
    model = Sequential(
        [
            Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # ... (no changes to this block)
    DATA_DIR = "./Training_Data/"
    X, y = load_data(DATA_DIR)
    if X.size == 0:
        print("No data loaded. Please check the 'Training_Data' directory. Exiting.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data shapes: Train={X_train.shape}, Test={X_test.shape}")
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y))
        print(
            f"Building model for {num_classes} classes with input shape {input_shape}"
        )
        model = build_model(input_shape, num_classes)
        model.summary()
        print("\n--- Starting Model Training ---")
        history = model.fit(
            X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test)
        )
        print("\n--- Evaluating Model Performance ---")
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        model.save("gesture_recognition_model.h5")
        print("Model saved to 'gesture_recognition_model.h5'")
