#!/usr/bin/env python3
"""
features/advanced_features.py

Advanced CSI feature extractors ported from provided MATLAB routines.

Functions:
  - naive_spectrum(csi: np.ndarray, fs: float, nperseg: int=None, noverlap: int=None)
  - naive_aoa(csi: np.ndarray, antenna_distances: list, fc: float)
  - delay_spread(csi: np.ndarray, bw: float)
  - spatial_correlation(csi: np.ndarray)

Each returns numpy arrays suitable for concatenation into a feature vector.
"""
import numpy as np
from scipy.signal import stft


def naive_spectrum(
    csi: np.ndarray, fs: float, nperseg: int = None, noverlap: int = None
):
    """
    Compute a Doppler spectrogram feature from CSI amplitude.

    Args:
        csi (np.ndarray): CSI data, shape (P, S, A, L)
        fs (float): sampling frequency in Hz (packets per second)
        nperseg (int): window length for STFT
        noverlap (int): overlap length for STFT

    Returns:
        np.ndarray: Spectrogram magnitude, shape (F, T)
    """
    # Aggregate amplitude across subcarriers and antennas
    seq = np.abs(csi).mean(axis=(1, 2, 3))  # shape: (P,)
    # Default window size
    if nperseg is None:
        nperseg = min(len(seq), 128)
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = stft(seq, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)


def naive_aoa(csi: np.ndarray, antenna_distances: list, fc: float):
    """
    Estimate Angle-of-Arrival (AoA) using phase differences between antennas.

    Args:
        csi (np.ndarray): CSI data, shape (P, S, A, L)
        antenna_distances (list): distances (m) between reference antenna and others, length A-1
        fc (float): center frequency in Hz

    Returns:
        np.ndarray: AoA estimates (radians), shape (P, A-1)
    """
    c = 3e8  # speed of light
    lam = c / fc
    P, S, A, L = csi.shape
    aoa = np.zeros((P, A - 1))
    # Use first TX chain (l=0) for phase
    for t in range(P):
        for a in range(1, A):
            ph0 = np.angle(csi[t, :, 0, 0])
            pha = np.angle(csi[t, :, a, 0])
            diff = np.unwrap(pha - ph0)
            mean_diff = np.mean(diff)
            d = antenna_distances[a - 1]
            aoa[t, a - 1] = np.arcsin((mean_diff * lam) / (2 * np.pi * d))
    return aoa


def delay_spread(csi: np.ndarray, bw: float):
    """
    Compute mean delay and RMS delay spread from CSI.

    Args:
        csi (np.ndarray): CSI data, shape (P, S, A, L)
        bw (float): total bandwidth in Hz

    Returns:
        np.ndarray: shape (P, 2) with [mean_delay, rms_spread] per packet
    """
    P, S, A, L = csi.shape
    dt = 1.0 / bw
    features = np.zeros((P, 2))
    for t in range(P):
        # average CSI magnitude across antennas and TX chains -> shape (S,)
        sub = np.abs(
            csi[t].mean(axis=(1, 2))
        )  # average over antennas and TX chains, shape (S,)
        # Channel impulse response via IFFT
        cir = np.fft.ifft(sub)
        mag2 = np.abs(cir) ** 2
        # Time vector for taps
        times = np.arange(len(cir)) * dt  # shape (S,)
        # Compute mean delay and RMS spread
        mean_delay = np.sum(times * mag2) / np.sum(mag2)
        rms = np.sqrt(np.sum((times - mean_delay) ** 2 * mag2) / np.sum(mag2))
        features[t, 0] = mean_delay
        features[t, 1] = rms
    return features


def spatial_correlation(csi: np.ndarray):
    """
    Compute spatial correlation coefficients between RX antenna pairs.

    Args:
        csi (np.ndarray): CSI data, shape (P, S, A, L)

    Returns:
        np.ndarray: shape (P, A*(A-1)/2) flattened upper-triangle correlations per packet
    """
    P, S, A, L = csi.shape
    # collapse subcarrier and tx dims
    chans = csi.reshape(P, S * L, A)  # shape (P, M, A)
    out = []
    for t in range(P):
        C = np.corrcoef(chans[t].T)  # A x A
        idx = np.triu_indices(A, k=1)
        out.append(C[idx])
    return np.vstack(out)
