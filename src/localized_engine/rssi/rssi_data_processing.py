#!/usr/bin/env python3
"""
compute_rssi_means.py

Read raw RSSI logs and compute per-point, per-anchor mean RSSI, with outlier clipping.
Outputs rssi_means.csv with columns: point_name,x,y,alias,mean_rssi
"""
import argparse
import logging
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-point mean RSSI from raw rssi_log.csv"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to raw RSSI log CSV (timestamp,point_name,x,y,alias,mac,rssi)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to write rssi_means.csv"
    )
    parser.add_argument(
        "--min-rssi", type=int, default=-100, help="Discard readings below this dBm"
    )
    parser.add_argument(
        "--max-rssi", type=int, default=0, help="Discard readings above this dBm"
    )
    return parser.parse_args()


def clip_outliers(series):
    """Clip values outside the 5th–95th percentile to reduce outlier impact."""
    lower, upper = series.quantile([0.05, 0.95])
    return series.clip(lower=lower, upper=upper)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger()

    logger.info(f"Loading raw RSSI data from {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        sys.exit(1)

    # Show available columns
    logger.info(f"Columns in input: {list(df.columns)}")

    # Detect RSSI column
    if "rssion" not in df.columns:
        candidates = [c for c in df.columns if "rssion" in c.lower()]
        if candidates:
            orig = candidates[0]
            df.rename(columns={orig: "rssion"}, inplace=True)
            logger.info(f"Renamed column '{orig}' to 'rssion'")
        else:
            logger.error("No RSSI column found in input.")
            sys.exit(1)

    # Filter impossible values
    df = df[(df["rssion"] >= args.min_rssi) & (df["rssion"] <= args.max_rssi)]
    logger.info(f"After filtering, {len(df)} samples remain")

    # Clip outliers per anchor
    if "alias" not in df.columns:
        logger.error("Missing 'alias' column in input.")
        sys.exit(1)
    df["rssi_clipped"] = df.groupby("alias")["rssion"].transform(clip_outliers)

    # Compute mean per point and alias
    group_cols = ["point_name", "x", "y", "alias"]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in input: {missing}")
        sys.exit(1)

    means = (
        df.groupby(group_cols, as_index=False)
        .rssi_clipped.mean()
        .rename(columns={"rssi_clipped": "mean_rssi"})
    )

    logger.info(f"Computed means for {len(means)} point–anchor pairs")
    try:
        means.to_csv(args.output, index=False)
        logger.info(f"Wrote rssi_means.csv to {args.output}")
    except Exception as e:
        logger.error(f"Failed to write output CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
