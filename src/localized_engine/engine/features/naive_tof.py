#!/usr/bin/env python3
"""
features/naive_tof.py

Naive Time-of-Flight (ToF) feature extractor for CSI data.

Provides:
  - naive_tof(csi: np.ndarray, bw: float) -> np.ndarray

Args:
  - csi: CSI data, shape (P, S, A, L)
  - bw: total bandwidth in Hz (e.g., 20e6)

Returns:
  - tof: shape (P, A), where tof[t, a] is the estimated time-of-flight (seconds) for packet t and RX antenna a.

Method:
  1. For each packet t and antenna a, average CSI over TX chains.
  2. Perform IFFT to get CIR.
  3. Find delay index of max amplitude tap and convert to time = index/bw.
"""
import numpy as np


def naive_tof(csi: np.ndarray, bw: float) -> np.ndarray:
    """
    Estimate time-of-flight (ToF) per packet and RX antenna using IFFT.

    Args:
        csi (np.ndarray): CSI data, shape (P, S, A, L)
        bw (float): Total subcarrier bandwidth in Hz

    Returns:
        np.ndarray: ToF array, shape (P, A)
    """
    P, S, A, L = csi.shape
    tof = np.zeros((P, A), dtype=float)
    # Sampling period per CIR tap
    dt = 1.0 / bw
    for t in range(P):
        for a in range(A):
            # average CSI across TX chains for this packet/antenna
            sub = csi[t, :, a, :].mean(axis=1)  # shape (S,)
            # Channel impulse response via IFFT
            cir = np.fft.ifft(sub)
            # Delay index of maximum magnitude
            idx = int(np.argmax(np.abs(cir)))
            # Convert index to time
            tof[t, a] = idx * dt
    return tof
