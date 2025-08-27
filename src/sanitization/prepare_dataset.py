#!/usr/bin/env python3
"""
data_processing/prepare_dataset.py

Prepare training dataset (features & labels) from raw CSI .dat files and ground truth positions.

Usage:
  python -m data_processing.prepare_dataset \
    --input-csv    data_files/widar_positions.csv \
    --calib-mat    data_files/calibration/csi_calib_test.mat \
    --linear-interval 20 39 \
    --features-out data_files/phase2/X.npy \
    --labels-out   data_files/phase2/y.npy

The input CSV should have columns:
  file_path,x,y

This script will:
  1. Load calibration CSI and build a template.
  2. For each row in CSV:
     - Replay CSI packets via ingest.replay_file()
     - Sanitize CSI with the template
     - Extract features using simple_amplitude + advanced extractors
     - Aggregate per-file features (mean across packets)
     - Append feature vector and corresponding (x,y) label
  3. Save the stacked feature matrix (N × F) and label matrix (N × 2) as .npy

"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
<<<<<<<< HEAD:dataset/prepare_dataset.py

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Localized_Engine import replay_file
from data_processing.sanitize_denoising import set_template
from data_processing.sanitize_denoising import sanitize_csi
from Localized_Engine.Trash.features import simple_amplitude

# Optionally import advanced features
from Localized_Engine.Trash.features import naive_aoa, delay_spread, spatial_correlation
from Localized_Engine.Trash.features import naive_tof

========
from localized_engine.engine.ingest.server import replay_file
from sanitization.sanitize_denoising import set_template, sanitize_csi
from localized_engine.engine.features.simple_amplitude import simple_amplitude

# Optionally import advanced features
try:
    from localized_engine.engine.features.advanced_features import naive_aoa, delay_spread, spatial_correlation
    from localized_engine.engine.features.advanced_features import naive_tof
except ImportError:
    naive_tof = naive_aoa = delay_spread = spatial_correlation = None
>>>>>>>> origin/second_prototype:src/sanitization/prepare_dataset.py


def parse_args():
    p = argparse.ArgumentParser(description="Prepare CSI feature/label datasets")
    p.add_argument(
        '--input-csv', required=True,
        help='CSV with columns: file_path,x,y'
    )
    p.add_argument(
        '--calib-mat', required=True,
        help='Path to calibration .mat file'
    )
    p.add_argument(
        '--linear-interval', nargs='+', type=int, required=True,
        help='Subcarrier indices for template creation (1-based)'
    )
    p.add_argument(
        '--features-out', required=True,
        help='Output .npy for features X'
    )
    p.add_argument(
        '--labels-out', required=True,
        help='Output .npy for labels y'
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Read mapping CSV
    df = pd.read_csv(args.input_csv)
    if df.empty:
        print(f"No entries found in {args.input_csv}")
        return

    # Build calibration CSI template from the provided calibration file
    if not os.path.isfile(args.calib_mat):
        raise FileNotFoundError(f"Calibration file not found: {args.calib_mat}")

    # Check file extension to determine how to load it
    file_ext = os.path.splitext(args.calib_mat)[1].lower()

    if file_ext == '.dat':
        # Use load_csi_data for .dat files
        from data_processing.sanitize_denoising import load_csi_data
        calib_arr, _ = load_csi_data(args.calib_mat)
    else:
        # Use loadmat for .mat files
        from scipy.io import loadmat
        mat = loadmat(args.calib_mat)
        # grab the first non‐private variable (assumed to be your CSI array)
        calib_arr = None
        for key, val in mat.items():
            if not key.startswith('__'):
                calib_arr = val
                break
        if calib_arr is None:
            raise RuntimeError(f"No usable CSI variable found in {args.calib_mat}")
    # ensure it’s 4D: (P, S, A, L)
    if calib_arr.ndim == 3:
        csi_calib = calib_arr[..., np.newaxis]
    else:
        csi_calib = calib_arr
    # clip your subcarrier indices to the actual range
    P, S, A, L = csi_calib.shape
    lin = np.array(args.linear_interval)
    valid = (lin >= 1) & (lin <= S)
    if not valid.all():
        print(f"⚠️  dropping out‐of‐range subcarriers: {lin[~valid]}")
    lin = lin[valid]
    linear = lin  # normalize name for downstream sanitization calls
    if lin.size < 2:
        raise ValueError(f"Need ≥2 valid subcarriers, got {lin.tolist()}")
    tpl = set_template(csi_calib, lin)

    X_list, y_list = [], []
    for _, row in df.iterrows():
        path = row['file_path']
        if not os.path.isfile(path):
            print(f"Warning: file not found {path}, skipping")
            continue
        packets = list(replay_file(path, endian='big'))
        if not packets:
            print(f"Warning: no packets from {path}, skipping")
            continue

        # Batch‐sanitize: stack raw CSI then sanitize in one call
        raw_csi = np.stack(packets)  # shape: (P, S, A, L) or (P, S, A)
        csi_arr = sanitize_csi(raw_csi, tpl, linear)  # sanitize handles 3D→4D internally

        # Extract features per packet
        feats = []
        feats.append(simple_amplitude(csi_arr).mean(axis=0))
        if naive_tof:
            feats.append(naive_tof(csi_arr, bw=20e6).mean(axis=0))
        if spatial_correlation:
            feats.append(spatial_correlation(csi_arr).mean(axis=0))
        if delay_spread:
            feats.append(delay_spread(csi_arr, bw=20e6).mean(axis=0))
        if naive_aoa:
            # Provide known antenna distances and fc
            feats.append(naive_aoa(csi_arr, antenna_distances=[0.05]*(csi_arr.shape[2]-1), fc=5e9).mean(axis=0))
        feature_vec = np.concatenate(feats)
        X_list.append(feature_vec)
        y_list.append([row['x'], row['y']])

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    os.makedirs(os.path.dirname(args.features_out), exist_ok=True)
    np.save(args.features_out, X)
    np.save(args.labels_out,   y)
    print(f"Saved features ({X.shape}) to {args.features_out}")
    print(f"Saved labels ({y.shape}) to {args.labels_out}")

if __name__ == '__main__':
    main()
