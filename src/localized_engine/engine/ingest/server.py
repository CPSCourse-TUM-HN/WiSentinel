#!/usr/bin/env python3
"""
ingest/server.py

Phase 2: CSI Ingestion Module

Provides a unified interface to replay CSI logs from Atheros or Intel 5300 formats.

Functions:
  - replay_file(datfile: str, endian: str = 'little') -> Iterator[np.ndarray]
      Attempts to read with Atheros parser (little/big endian), and falls back to Intel parser.
      Yields one CSI array per packet, shape (subcarriers, rx, tx).
"""
from typing import Iterator
import csiread
import numpy as np


def replay_file(datfile: str, endian: str = "little") -> Iterator[np.ndarray]:
    """
    Replay a CSI .dat file, auto-detecting between Atheros and Intel formats.

    Args:
        datfile: Path to the .dat file
        endian: Byte order ('little' or 'big') for Atheros reader

    Yields:
        CSI arrays per packet: ndarray of shape (subcarriers, rx, tx)
    """
    # Try Atheros reader
    try:
        ath = csiread.Atheros(datfile)
        ath.read(endian=endian)
        csi_all = ath.csi  # numpy array (P, S, A, L) complex
    except Exception:
        # Fallback to Intel reader
        intel = csiread.Intel(datfile)
        intel.read()
        csi_all = intel.csi  # numpy array
    # Yield each packet individually
    for pkt in csi_all:
        yield pkt
