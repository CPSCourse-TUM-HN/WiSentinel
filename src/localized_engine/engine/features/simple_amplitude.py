#!/usr/bin/env python3
"""
features/simple_amplitude.py

Simple amplitude feature extractor for CSI data.

Provides:
  - simple_amplitude(csi: np.ndarray) -> np.ndarray

Where:
  - csi: raw or sanitized CSI, shape (P, S, A, L)
  - returns: feature matrix X of shape (P, F), where
      F = S * A * L and X[p] is the flattened mean amplitude across packets if desired.
"""
import numpy as np


def simple_amplitude(csi: np.ndarray) -> np.ndarray:
    """
    Compute simple amplitude features by taking the mean amplitude per packet
    and flattening subcarrier/antenna dimensions.

    Args:
        csi (np.ndarray): CSI data, shape (P, S, A, L)

    Returns:
        np.ndarray: Features, shape (P, S*A*L)
    """
    # Compute amplitude
    amp = np.abs(csi)  # shape: (P, S, A, L)
    # Mean over packet dimension? Actually keep per-packet features
    # Flatten per packet
    P, S, A, L = amp.shape
    # Flatten S, A, L
    X = amp.reshape(P, S * A * L)
    return X
