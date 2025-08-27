#!/usr/bin/env python3
"""
CSI Preprocessing – Phase 1 (Simple)

This simplified script reads raw Atheros CSI `.dat` files, computes amplitude, and saves it for Phase 2.

Usage:
  python csi_preprocessing_phase1_simple.py \
    --input  data_files/csi_data/ath_csi_1.dat \
    --endian little \
    --output data_files/processed/csi_amp.npz

Outputs:
  - csi_amp.npz with array `csi_amp` (shape: [packets, rx, tx, subcarriers])

Phase 1 Deliverable:
  - Amplitude-only CSI ready for feature extraction and model input.
"""
import argparse
import logging
import os
import numpy as np
import csiread


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: CSI amplitude extraction")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to raw CSI .dat file"
    )
    parser.add_argument(
        "--endian",
        "-e",
        choices=["little", "big"],
        default="little",
        help="Endianness for CSI data reading",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save CSI amplitude (.npz)"
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger()


def main():
    args = parse_args()
    logger = setup_logging()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Read raw CSI
    logger.info(f"Reading raw CSI from {args.input}")
    csidata = csiread.Atheros(
        args.input, ntxnum=3
    )  # Explicitly set ntxnum=3 to handle files with more transmit antennas
    csidata.read(endian=args.endian)
    csi = csidata.csi
    logger.info(f"Loaded CSI shape={csi.shape}, dtype={csi.dtype}")

    # Compute amplitude
    csi_amp = np.abs(csi)
    logger.info(f"Computed amplitude: shape={csi_amp.shape}")

    # Save amplitude
    logger.info(f"Saving CSI amplitude to {args.output}")
    np.savez_compressed(args.output, csi_amp=csi_amp)
    logger.info("CSI amplitude saved.")


if __name__ == "__main__":
    main()
