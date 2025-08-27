#!/usr/bin/env python3
"""
rssi_localization.py

Industry-grade script for RSSI-only indoor localization pilot (Phase B).
Supports:
  - Path-loss model trilateration
  - KNN fingerprinting (regression & classification)

Usage:
  python rssi_localization.py \
    --anchors anchors.json \
    --means rssi_means.csv \
    --mode {trilat,fingerprint} \
    [-k 3] \
    [--zone-map zones.csv] \
    [--output-prefix results]

Outputs:
  <prefix>_trilat_metrics.json or <prefix>_fingerprint_metrics.json
  commit issues addressed -- Rebased on latest RSSI_Data_Processing.py
"""
import argparse
import json
import logging
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_anchors(path: Path) -> dict:
    """Load anchor positions from JSON array [{alias, coords}]."""
    try:
        data = json.load(open(path))
    except Exception as e:
        logger.error(f"Failed to load anchors JSON {path}: {e}")
        sys.exit(1)
    anchors = {}
    for a in data:
        alias = a.get("alias")
        coords = a.get("coords")
        if alias is None or coords is None or len(coords) != 2:
            logger.error(f"Invalid anchor entry: {a}")
            sys.exit(1)
        anchors[alias] = (float(coords[0]), float(coords[1]))
    logger.info(f"Loaded {len(anchors)} anchors from {path}")
    return anchors


def load_means(path: Path) -> pd.DataFrame:
    """Load per-point mean RSSI CSV with columns: point_name,x,y,alias,mean_rssi"""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read means CSV {path}: {e}")
        sys.exit(1)
    required = {"point_name", "x", "y", "alias", "mean_rssi"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Means CSV missing columns: {missing}")
        sys.exit(1)
    logger.info(f"Loaded {len(df)} RSSI mean records from {path}")
    return df


def calibrate_path_loss(df: pd.DataFrame, anchors: dict) -> tuple:
    """Calibrate path-loss model P0,n from means and known anchors."""
    data = []
    for _, row in df.iterrows():
        alias = row["alias"]
        if alias not in anchors:
            continue
        ax, ay = anchors[alias]
        dx = row["x"] - ax
        dy = row["y"] - ay
        d = np.hypot(dx, dy)
        if d <= 0:
            continue
        data.append((d, row["mean_rssi"]))
    if not data:
        logger.error("No valid calibration data; check anchors vs means.")
        sys.exit(1)
    dists, rssis = map(np.array, zip(*data))

    def model(d, P0, n):
        return P0 - 10 * n * np.log10(d)

    popt, _ = curve_fit(model, dists, rssis, p0=[-30, 2])
    P0, n = popt
    logger.info(f"Calibrated path-loss: P0={P0:.2f} dBm, n={n:.2f}")
    return P0, n


def rssi_to_dist(rssi: float, P0: float, n: float) -> float:
    """Convert RSSI to distance using the calibrated model."""
    return 10 ** ((P0 - rssi) / (10 * n))


def trilaterate_point(anchors_xy: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Solve 2D position from multiple distances."""
    x1, y1 = anchors_xy[0]
    d1 = ds[0]
    A, b = [], []
    for (xi, yi), di in zip(anchors_xy[1:], ds[1:]):
        A.append([2 * (xi - x1), 2 * (yi - y1)])
        b.append(xi * xi + yi * yi - di * di - (x1 * x1 + y1 * y1 - d1 * d1))
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return sol


def run_trilateration(df: pd.DataFrame, anchors: dict, prefix: Path):
    P0, n = calibrate_path_loss(df, anchors)
    points = df[["point_name", "x", "y"]].drop_duplicates().reset_index(drop=True)
    alias_list = list(anchors.keys())
    coords = np.array([anchors[a] for a in alias_list])
    errors, times = [], []
    for _, pt in points.iterrows():
        sub = df[df.point_name == pt.point_name]
        rssis = sub.set_index("alias").loc[alias_list, "mean_rssi"].values
        start = time.time()
        ds = np.array([rssi_to_dist(r, P0, n) for r in rssis])
        est = trilaterate_point(coords, ds)
        times.append(time.time() - start)
        true = np.array([pt.x, pt.y])
        errors.append(np.linalg.norm(est - true))
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    latency = float(np.mean(times))
    logger.info(
        f"Trilat MAE={mae:.2f}m, RMSE={rmse:.2f}m, latency={latency*1000:.1f}ms"
    )
    metrics = {"mae_m": mae, "rmse_m": rmse, "latency_s": latency}

    out_file = prefix.parent / f"{prefix.name}_trilat_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved trilat metrics to {out_file}")


def run_fingerprint(
    df: pd.DataFrame, anchors: dict, zones: pd.DataFrame, k: int, prefix: Path
):
    feat = df.pivot(index="point_name", columns="alias", values="mean_rssi")
    meta = df[["point_name", "x", "y"]].drop_duplicates().set_index("point_name")
    X = feat.values
    Y = meta[["x", "y"]].values
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.3, random_state=42)
    X_train, X_test = X[tr], X[te]
    y_train, y_test = Y[tr], Y[te]
    start = time.time()
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_train, y_train)
    preds = knn_reg.predict(X_test)
    latency = (time.time() - start) / len(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    logger.info(
        f"KNN-Reg MAE={mae:.2f}m, RMSE={rmse:.2f}m, latency={latency*1000:.1f}ms"
    )
    metrics = {"reg_mae_m": mae, "reg_rmse_m": rmse, "reg_latency_s": latency}
    if zones is not None:
        zm = zones.set_index("point_name").loc[feat.index]
        labels = zm["zone"].values
        y_tr_z, y_te_z = labels[tr], labels[te]
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        knn_clf.fit(X_train, y_tr_z)
        acc = accuracy_score(y_te_z, knn_clf.predict(X_test))
        metrics["zone_accuracy"] = acc
        logger.info(f"KNN-Clf Zone Acc={acc*100:.1f}%")
    out_file = prefix.parent / f"{prefix.name}_fingerprint_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved fingerprint metrics to {out_file}")


def main():
    p = argparse.ArgumentParser(
        description="RSSI Localization: Trilateration & Fingerprint"
    )
    p.add_argument("--anchors", type=Path, required=True, help="Anchors JSON")
    p.add_argument("--means", type=Path, required=True, help="RSSI means CSV")
    p.add_argument("--mode", choices=["trilat", "fingerprint"], required=True)
    p.add_argument("-k", type=int, default=3, help="K for KNN")
    p.add_argument("--zone-map", type=Path, help="CSV: point_name,zone")
    p.add_argument(
        "--output-prefix", type=Path, default=Path("results"), help="Output file prefix"
    )
    args = p.parse_args()
    anchors = load_anchors(args.anchors)
    df_means = load_means(args.means)
    zones = pd.read_csv(args.zone_map) if args.zone_map else None
    if args.mode == "trilat":
        run_trilateration(df_means, anchors, args.output_prefix)
    else:
        run_fingerprint(df_means, anchors, zones, args.k, args.output_prefix)


if __name__ == "__main__":
    main()
