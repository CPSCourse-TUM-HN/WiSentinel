"""
Script to extract features from CSI .dat files using the Atheros class and FeatureExtractor.
- Removes NaN packets and checks the shape of each packet.
- Computes all available features using FeatureExtractor.
- Saves processed features to output files in the feature_data folder.
- Data is split into train/test/validation (default 70%/15%/15%) and saved in corresponding subfolders.
- Features are calculated using windows of 60 packets.
- For 'standing' files, first and last 150 packets are cut off.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ====== CONFIGURABLE SPLIT RATIOS ======
TRAIN_RATIO = 0.7
TEST_RATIO = 0.10
VAL_RATIO = 0.20
# =======================================

# Import Atheros CSI reader and FeatureExtractor
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from csiread_lib.atheros_csi_read import Atheros
sys.path.append(str(Path(__file__).resolve().parents[2]))
from feature_extraction import FeatureExtractor
from sanitization import SanitizeDenoising
from sanitization import load_csi_data

WINDOW_SIZE = 30
CORRECT_SHAPE = (56, 3, 3)
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

def find_dat_files(root_folder):
    """Recursively find all .dat files in the root_folder."""
    dat_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".dat"):
                dat_files.append(os.path.join(dirpath, fname))
    return dat_files

def split_indices(n, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    """
    Return indices for train, test, validation splits.
    Guarantees at least one sample in each split if possible.
    """
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Calculate split sizes
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test

    # Guarantee at least 1 in each split if possible
    splits = [n_train, n_test, n_val]
    names = ['train', 'test', 'validation']
    for i in range(3):
        if splits[i] == 0 and n >= 3:
            splits[i] = 1

    # Adjust so that sum(splits) == n
    while sum(splits) > n:
        for i in range(3):
            if splits[i] > 1 and sum(splits) > n:
                splits[i] -= 1
    while sum(splits) < n:
        for i in range(3):
            if sum(splits) < n:
                splits[i] += 1

    n_train, n_test, n_val = splits

    # Assign indices
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    val_idx = indices[n_train + n_test:]
    return train_idx, test_idx, val_idx

def process_csi_file(dat_file, feature_extractor):
    """
    Read a .dat file, sanitize it, and extract per-packet features.
    Returns: DataFrame of shape [n_packets, feature_count]
    """
    # 1. Read CSI data
    csi_reader = Atheros(dat_file)
    csi_reader.read()
    csi_reader.remove_nan_packets()
    csi = csi_reader.csi

    if any(pose in dat_file for pose in [os.sep + "standing" + os.sep, os.sep + "crouching" + os.sep]):
        if len(csi) > 300:
            csi = csi[150:-150]
        else:
            print(f"[WARN] Not enough packets to cut off for {dat_file}. Skipping.")
            return None

    if len(csi) < 1:
        print(f"[WARN] No valid packets in {dat_file}. Skipping.")
        return None

    # Calibration for sanitization
    calib_path = "../dataset/ilive-2/no_person/ilive-2-empty-room-1min.dat"
    calib_csi, _ = load_csi_data(calib_path)
    sanitizer = SanitizeDenoising(calib_path)
    linear_interval = np.arange(20, 39)
    template = sanitizer.set_template(calib_csi, linear_interval)

    sanitized_csi = sanitizer.sanitize_csi_selective(
        csi, template, nonlinear=True, sto=True, rco=True, cfo=False
    )
    print(f"Sanitized packets shape: {len(sanitized_csi)}")

    # 2. Stack and extract per-packet features
    csi_valid = np.stack(sanitized_csi, axis=0)
    features_df = feature_extractor.extract_features_sequence(csi_valid)

    # 3. Add label
    if "no_person" in dat_file:
        label = "no_person"
    elif "standing" in dat_file:
        label = "standing"
    elif "crouching" in dat_file:
        label = "crouching"
    else:
        label = "unknown"

    features_df["label"] = label
    features_df["source_file"] = os.path.basename(dat_file)
    return features_df


def save_split_features_per_file(features_df, out_base_folder, pose_folder, base_filename):
    """Save features DataFrame to CSV in the appropriate split subfolders for each file."""
    n_samples = len(features_df)
    train_idx, test_idx, val_idx = split_indices(n_samples)
    splits = {
        "train": features_df.iloc[train_idx],
        "test": features_df.iloc[test_idx],
        "validation": features_df.iloc[val_idx],
    }
    for split_name, split_df in splits.items():
        split_folder = os.path.join(out_base_folder, pose_folder, split_name)
        os.makedirs(split_folder, exist_ok=True)
        out_path = os.path.join(split_folder, f"{base_filename}_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved {split_name} features to {out_path}")

def main():
    # 1. Set up paths
    data_root = "../dataset/ilive-2"
    feature_data_dir = "feature_data_2"
    os.makedirs(feature_data_dir, exist_ok=True)

    # 2. Find all .dat files in the data directory
    dat_files = find_dat_files(data_root)
    print(f"[INFO] Found {len(dat_files)} .dat files.")

    # 3. Initialize feature extractor
    feature_extractor = FeatureExtractor(FEATURE_LIST)

    # 4. Process each .dat file and save splits per file
    for dat_file in dat_files:
        print(f"[INFO] Processing {dat_file}")
        features_df = process_csi_file(dat_file, feature_extractor)
        if features_df is None:
            continue

        # Determine pose folder (relative to data_root)
        rel_path = os.path.relpath(dat_file, data_root)
        pose_folder = rel_path.split(os.sep)[0]
        base_filename = os.path.splitext(os.path.basename(dat_file))[0]

        save_split_features_per_file(features_df, feature_data_dir, pose_folder, base_filename)

    print("[INFO] Feature extraction and saving complete.")

if __name__ == "__main__":
    main()