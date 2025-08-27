import os
import joblib
import numpy as np
import pygame  # The core library for our visualizer
from collections import Counter
import logging
from collections import deque
import hashlib
import pandas as pd
from tensorflow.keras.models import load_model

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from feature_extraction import FeatureExtractor

logging.basicConfig(level=logging.DEBUG)

from sanitization import SanitizeDenoising, load_csi_data
from csiread_lib import AtherosUDPRealtime

def summarize_csi_stack(csi_stack: np.ndarray) -> str:
    """Generate a hash + stats summary of a CSI window."""
    flat_bytes = csi_stack.tobytes()
    checksum = hashlib.md5(flat_bytes).hexdigest()[:8]  # short hash
    magnitude = np.abs(csi_stack)
    stats = {
        "hash": checksum,
        "min": np.min(magnitude),
        "max": np.max(magnitude),
        "mean": np.mean(magnitude),
        "nan_count": np.isnan(magnitude).sum(),
        "inf_count": np.isinf(magnitude).sum(),
    }
    return (
        f"[CSI window] hash={stats['hash']} | "
        f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
        f"mean={stats['mean']:.2f}, NaNs={stats['nan_count']}, Infs={stats['inf_count']}"
    )


# --- Configuration: All required assets and settings ---


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 1. Asset Paths
CALIBRATION_FILE_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'ilive-2', 'no_person', 'ilive-2-empty-room-1min.dat')

# 2. Visualizer Assets
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, '..', 'images', 'floorplan.png')
PERSON_ICON_PATH = os.path.join(BASE_DIR, '..', 'images', 'person_icon.png')

# 3. Screen and Room Dimensions for Coordinate Mapping
#    **IMPORTANT**: You MUST adjust these values to match your room and image.
SCREEN_WIDTH = 800  # pixels
SCREEN_HEIGHT = 600  # pixels

# Physical dimensions of the room IN METERS
ROOM_WIDTH_M = 5.0  # e.g., 5 meters wide
ROOM_HEIGHT_M = 3.75  # e.g., 3.75 meters tall

PORT_NUMBER = 8000  # Port for Atheros UDP packets
WINDOW_SIZE = 30  # Number of packets to average over
REFRESH_RATE = 1  # How often to update the visualizer (in seconds)


FEATURE_LIST = [
    "tof_mean",
    "tof_std",
    "aoa_peak",
    "corr_eigen_ratio",
    "num_peaks",
    "std_amp",
    "median_amp",
    "skew_amp",
    "kurtosis_amp",
]

PRESENCE_FEATURES = [
    'tof_mean', 'tof_std', 'corr_eigen_ratio',
    'num_peaks', 'std_amp', 'median_amp', 'skew_amp', 'kurtosis_amp'
]

COORDINATES_FEATURES = ['tof_mean', 'tof_std', 'aoa_peak', 'corr_eigen_ratio', 'std_amp', 'median_amp', 'skew_amp']

# --- Helper Functions (from previous phases) ---

# Maintain persistent state
csi_window = deque(maxlen=WINDOW_SIZE)
last_coords = None


def get_prediction(
    sanitizer, template, 
    human_detector, pose_classifier, coordinates_regressor,
    feature_extractor, udp_reader
):
    global last_coords

    try:
        packets = udp_reader.drain_buffer()
        for raw_csi in packets:
            if raw_csi is None:
                continue
            raw_csi = raw_csi[None, :, :, :]
            sanitized = sanitizer.sanitize_csi_selective(raw_csi, np.clip(template, 1e-3, None))
            if sanitized is None:
                continue
            csi_window.append(sanitized[0])

        if len(csi_window) < WINDOW_SIZE:
            return False, None, None

        # Sanitize full window
        csi_array = np.stack(csi_window)
        sanitized_csi = sanitizer.sanitize_csi_selective(
            csi_array, template, nonlinear=True, sto=True, rco=True, cfo=False
        )

        # Extract features
        feature_df = feature_extractor.extract_features_sequence(sanitized_csi)

        # Prepare inputs per model
        X_presence = feature_df[PRESENCE_FEATURES].values.astype(np.float32)
        X_coords   = feature_df[COORDINATES_FEATURES].values.astype(np.float32)

        X_presence = np.expand_dims(X_presence, axis=0)  # (1, 30, Fp)
        X_coords   = np.expand_dims(X_coords, axis=0)    # (1, 30, Fc)

        # Human detection
        presence_prob = human_detector.predict(X_presence)[0][0]
        presence = presence_prob > 0.5
        logging.info(f"[PRESENCE] Probability: {presence_prob:.2%}")

        if not presence:
            logging.info("[🪑] No human detected.")
            last_coords = None
            return False, None, None

        # Coordinates
        coords = coordinates_regressor.predict(X_coords)[0]
        coords = np.clip(coords, 0.0, 2.0)  # Clamp hallucinations to room range

        # Pose
        pose_probs = pose_classifier.predict(X_presence)
        pose_idx = int(np.argmax(pose_probs))
        pose_label = "crouching" if pose_idx == 1 else "standing"

        logging.info(f"[✅] Human detected at coords=({coords[0]:.2f}, {coords[1]:.2f}), pose={pose_label}")
        last_coords = coords
        return True, (coords[0], coords[1]), pose_label

    except Exception as e:
        logging.exception(f"[ERROR] Exception during prediction: {e}")
        return False, None, None




def transform_coords(physical_x, physical_y):
    """
    Transforms physical meter coordinates to screen pixel coordinates.
    This function is CRITICAL for correctly positioning the icon.
    """
    # Calculate pixels per meter
    pixels_per_meter_x = SCREEN_WIDTH / ROOM_WIDTH_M
    pixels_per_meter_y = SCREEN_HEIGHT / ROOM_HEIGHT_M

    pixel_x = int(physical_x * pixels_per_meter_x)

    pixel_y = int(physical_y * pixels_per_meter_y)

    return pixel_x, pixel_y

def execute_visualizer():
    print("🚀 INITIALIZING VISUALIZER...")

    try:
        # Load models
        # Load models
        human_detector = load_model("train_models/human_detection/human_detection_cnn_model.h5", compile=False)
        coordinates_regressor = load_model("train_models/coordinates_estimation/coordinates_cnn_model.h5", compile=False)
        pose_classifier = load_model("train_models/pose_estimation/pose_estimation_cnn_model.h5", compile=False)


        # Feature extractor
        feature_extractor = FeatureExtractor(FEATURE_LIST)

        # Sanitizer and UDP
        sanitizer = SanitizeDenoising(CALIBRATION_FILE_PATH)
        linear_interval = np.arange(20, 39)

        calib_csi, _ = load_csi_data(CALIBRATION_FILE_PATH)
        template = sanitizer.set_template(calib_csi, linear_interval)

        udp_reader = AtherosUDPRealtime(port=PORT_NUMBER, log_level=logging.INFO)
        udp_reader.start()
        logging.info("UDP CSI reader initialized")
    except Exception as e:
        print(f"[FATAL ERROR] Could not initialize assets: {e}")
        return

    # 2. Initialize Pygame and screen elements
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Wi-Fi Localization Engine")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)  # Default font, size 36

    # Load and scale graphics
    try:
        background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
        background_img = pygame.transform.scale(background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        person_icons = {
            "standing": pygame.transform.scale(
                pygame.image.load(os.path.join(BASE_DIR, '..', 'images', 'person_standing.png')).convert_alpha(), (200, 200)
            ),
            "crouching": pygame.transform.scale(
                pygame.image.load(os.path.join(BASE_DIR, '..', 'images', 'person_crouching.png')).convert_alpha(), (200, 200)
            )
    }
    except pygame.error as e:
        print(f"[FATAL ERROR] Could not load image assets. Make sure they are in the directory: {e}")
        return

    print("[SUCCESS] Initialization complete. Starting real-time visualizer.")

    # 3. The Main Real-Time Loop
    running = True
    while running:
        # Event handling (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get the latest prediction from our engine
        is_present, coords, pose = get_prediction(
            sanitizer, template,
            human_detector, pose_classifier, coordinates_regressor,
            feature_extractor, udp_reader
        )

        # --- Drawing Logic ---
        # 1. Draw the background (clears the screen)
        screen.blit(background_img, (0, 0))

        # 2. If a person is detected, draw them
        if is_present:
            # Transform coordinates to screen
            icon_center_x, icon_center_y = transform_coords(coords[0], coords[1])

            # Adjust icon position
            icon_pos_x = icon_center_x - 25
            icon_pos_y = icon_center_y - 25

            # Choose icon by pose
            if pose in person_icons:
                pose_icon = person_icons[pose]
                screen.blit(pose_icon, (icon_pos_x, icon_pos_y))
            else:
                logging.warning(f"[WARN] Unknown pose label '{pose}'")

            # Draw coordinates as text
            text = f"{pose.capitalize()} ({coords[0]:.2f}m, {coords[1]:.2f}m)"
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (icon_pos_x, icon_pos_y - 30))


        # 3. Update the entire screen
        pygame.display.flip()

        # 4. Control the frame rate to ~1 update per second
        clock.tick(1)

    pygame.quit()
    udp_reader.stop()
    print("[INFO] Visualizer stopped.")


if __name__ == '__main__':
    execute_visualizer()