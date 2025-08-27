#!/usr/bin/env python3
"""

Don't use this anchor_calibration.py if anchors point are too close to each other, gives very unusal coordianates.
In anchors_est.json, I used the manual way to locate anchors, which is more accurate.
The other future work is to create a zone map, but the thing is that applies if the system is intstalled in multi-room environment.
anchor_calibration.py

Automatically estimate 2D router (anchor) positions from per-point RSSI measurements.
Uses path-loss model and non-linear least squares to self-calibrate anchors.

Usage:
  python anchor_calibration.py \
    --means rssi_means.csv \
    --output anchors_est.json

Outputs:
  anchors_est.json with JSON array [{"alias":..., "coords":[x,y]}, ...]
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Anchor self-calibration from RSSI means")
    p.add_argument(
        "--means", type=Path, required=True, help="CSV: point_name,x,y,alias,mean_rssi"
    )
    p.add_argument(
        "--output", type=Path, required=True, help="Output JSON for estimated anchors"
    )
    return p.parse_args()


def calibrate_path_loss(global_df):
    """
    Calibrate global path-loss P0 and exponent n using all anchors and points.
    Returns (P0, n).
    """
    # Prepare data arrays
    dists = []
    rssis = []
    # Temporary need anchors positions? Instead use distances approximated by neighbor points
    # Here, we require initial approximate distances for calibration. For self-calibration,
    # we assume path-loss exponent first by taking inter-anchor relative distances.
    # For simplicity, fit P0,n by treating mean_rssi ~ P0 - 10 n log10(d),
    # using distances from unknown anchors replaced by relative distances between points.
    # As a workaround, use nearest neighbor distances: the smallest and largest inter-point dims
    coords = global_df[["x", "y"]].drop_duplicates().values
    # Use grid spacing ~2m for approximate d0
    d0 = 1.0
    # Use two representative RSSI values at d0 and at max distance
    df0 = global_df.sample(100, replace=True)
    for _, row in df0.iterrows():
        rssis.append(row["mean_rssi"])
        dists.append(d0)
    # Fit trivial model with P0 = mean(rssi), n=2
    P0 = np.mean(rssis)
    n = 2.0
    logger.info(f"Assumed path-loss parameters: P0={P0:.2f}, n={n:.2f}")
    return P0, n


def objective_anchor(pos, xi, di):
    """Residuals for anchor at position pos given measurement points xi and distances di."""
    return np.linalg.norm(xi - pos, axis=1) - di


def main():
    args = parse_args()

    # Load mean RSSI data
    df = pd.read_csv(args.means)
    required = ["x", "y", "alias", "mean_rssi"]
    if not all(c in df.columns for c in required):
        logger.error(f"Input CSV must contain columns: {required}")
        return

    # Unique anchors and points
    anchors = df["alias"].unique()
    points = df[["point_name", "x", "y"]].drop_duplicates()
    logger.info(f"Found {len(anchors)} anchors and {len(points)} survey points")

    # Calibrate global path-loss parameters
    # For simplicity, we’ll ask user to provide P0,n manually if accuracy needed
    P0, n = calibrate_path_loss(df)

    # Estimate anchor positions
    est_anchors = []
    for alias in anchors:
        sub = df[df["alias"] == alias]
        xi = sub[["x", "y"]].values  # survey point coords
        # Convert RSSI to distances
        di = 10 ** ((P0 - sub["mean_rssi"].values) / (10 * n))
        # Initial guess: centroid of survey points
        init = xi.mean(axis=0)
        res = least_squares(objective_anchor, x0=init, args=(xi, di))
        est = res.x.tolist()
        est_anchors.append({"alias": alias, "coords": est})
        logger.info(f"Anchor '{alias}' estimated at x={est[0]:.2f}, y={est[1]:.2f}")

    # Save results
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(est_anchors, f, indent=2)
    logger.info(f"Saved estimated anchors to {args.output}")


if __name__ == "__main__":
    main()
