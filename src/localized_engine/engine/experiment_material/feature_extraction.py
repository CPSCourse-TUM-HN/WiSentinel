#!/usr/bin/env python3
"""
feature_extraction_phase2.py

Phase 2: Extract feature & range labels from sanitized CSI amplitude.

Usage:
  python feature_extraction_phase2.py \
    --csi    data_files/processed/csi_amp.npz \
    --anchors anchors_est.json \
    --grid   grid_points.csv \
    --outdir data_files/phase2

Outputs:
  - X.npy (shape: [num_points, features])
  - y.npy (shape: [num_points, num_anchors])
"""
import argparse
import json
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csi", required=True, help="NPZ file from Phase1 with csi_amp")
    p.add_argument("--anchors", required=True, help="anchors_est.json")
    p.add_argument("--grid", required=True, help="grid_points.csv")
    p.add_argument("--outdir", required=True, help="output directory for .npy files")
    return p.parse_args()


def compute_distance(x, y, anchor):
    return np.hypot(x - anchor[0], y - anchor[1])


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load CSI amplitude
    data = np.load(args.csi)
    if "csi" in data:
        arr = data["csi"]
    elif "csi_amp" in data:
        arr = data["csi_amp"]
    else:
        raise KeyError(f"Could not find 'csi' or 'csi_amp' in {args.csi}")
    # arr shape: (P, rx, tx, S)
    amp = arr  # already amplitude

    # Load grid and anchors
    grid = pd.read_csv(args.grid)  # columns: x,y,point_name
    with open(args.anchors) as f:
        anchors_list = json.load(f)
    anchor_coords = {a["alias"]: tuple(a["coords"]) for a in anchors_list}

    # Determine per-point segment length
    P = amp.shape[0]
    M = len(grid)
    if P % M != 0:
        raise ValueError(f"Total packets {P} not divisible by grid points {M}")
    per_point = P // M

    X, y = [], []
    aliases = list(anchor_coords.keys())

    for i, row in grid.iterrows():
        segment = amp[i * per_point : (i + 1) * per_point]  # (per_point, rx, tx, S)
        # Feature: mean amplitude across packets, then flatten
        feat = segment.mean(axis=0).flatten()
        X.append(feat)
        # True distances
        ds = [compute_distance(row.x, row.y, anchor_coords[a]) for a in aliases]
        y.append(ds)

    X = np.stack(X)
    y = np.stack(y)

    np.save(os.path.join(args.outdir, "X.npy"), X)
    np.save(os.path.join(args.outdir, "y.npy"), y)
    print(f"Saved X.npy ({X.shape}) and y.npy ({y.shape}) to {args.outdir}")


if __name__ == "__main__":
    main()
