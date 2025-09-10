import numpy as np
from collections import deque
import pandas as pd

# --- Feature Extraction Imports ---
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from scipy.signal import stft


# Feature extraction functions for CSI data

class FeatureExtractor:
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def extract_features_window(self, window: np.ndarray) -> pd.DataFrame:
        """
        Extract features from a CSI window.
        window: np.ndarray of shape (N_packets, 56, 3, 3)
        Returns: DataFrame with extracted features (1 row)
        """
        feature_values = {}

        for feat_name in self.feature_list:
            method_name = f"{feat_name}_window"

            if not hasattr(self, method_name):
                raise ValueError(f"Feature method '{method_name}' is not implemented.")

            method = getattr(self, method_name)
            result = method(window)

            # For features returning multiple values (like dicts), update the dictionary
            if isinstance(result, dict):
                feature_values.update(result)
            else:
                feature_values[feat_name] = result

        return pd.DataFrame([feature_values])
        

# ============ Feature Extraction Functions Custom ============

# Differential amplitude across antenna pairs
    def amplitude_diff_packet(self, csi_packet):
        """
        csi_packet: np.ndarray of shape (56, 3, 3)
        Returns a feature: std of avg amplitudes across rx/tx
        """
        amp = np.abs(csi_packet)  # (56, 3, 3)
        mean_amp_per_pair = np.mean(amp, axis=0)  # (3, 3)
        diff_std = np.std(mean_amp_per_pair)  # how unequal are the antenna pairs?
        return diff_std


    # Variance over time in each 3D slot
    def temporal_variance_window(self, csi_window: np.ndarray):
        """
        csi_window: np.ndarray of shape (N, 56, 3, 3)
        Returns: dict of {f"var_rx{i}_tx{j}": value}
        """
        assert csi_window.ndim == 4  # (N_packets, 56, rx, tx)
        features = {}
        N, S, RX, TX = csi_window.shape
        
        for rx in range(RX):
            for tx in range(TX):
                magnitudes = np.abs(csi_window[:, :, rx, tx])  # (N, 56)
                var_over_time = np.var(magnitudes, axis=0).mean()  # mean variance over subcarriers
                features[f"var_rx{rx}_tx{tx}"] = var_over_time
        return features


    # ============= Feature Extraction Functions From Widar 3.0 Paper =============


    # ============= 9 Feature Extraction Functions Packet =============
    def extract_features_sequence(self, window: np.ndarray) -> pd.DataFrame:
        """Returns one feature vector per packet in window."""
        feature_rows = []
        for packet in window:
            feature_row = {}
            for feat_name in self.feature_list:
                method_name = f"{feat_name}_packet"
                if not hasattr(self, method_name):
                    raise ValueError(f"Feature method '{method_name}' is not implemented.")
                result = getattr(self, method_name)(packet)
                if isinstance(result, dict):
                    feature_row.update(result)
                else:
                    feature_row[feat_name] = result
            feature_rows.append(feature_row)
        return pd.DataFrame(feature_rows)


    def tof_mean_packet(self, csi_packet):
        cir = np.fft.ifft(csi_packet, axis=0)
        cir_abs = np.abs(cir)
        tof_index = np.argmax(cir_abs, axis=0)
        return np.mean(tof_index)

    def tof_std_packet(self, csi_packet):
        cir = np.fft.ifft(csi_packet, axis=0)
        cir_abs = np.abs(cir)
        tof_index = np.argmax(cir_abs, axis=0)
        return np.std(tof_index)

    def aoa_peak_packet(self, csi_packet):
        num_antennas = csi_packet.shape[1]
        if num_antennas < 2:
            return 0
        csi_avg_subcarriers = np.mean(csi_packet, axis=0)
        R = np.cov(csi_avg_subcarriers.T)
        eig_vals, eig_vecs = np.linalg.eigh(R)
        noise_subspace = eig_vecs[:, :-1]
        angles = np.linspace(-90, 90, 181)
        c = 3e8
        freq_mhz = 2437
        ant_dist_m = 0.05
        wavelength = c / (freq_mhz * 1e6)
        spectrum = []
        for angle in angles:
            steering_vector = np.exp(-1j * 2 * np.pi * ant_dist_m * np.arange(num_antennas)
                                     * np.sin(np.deg2rad(angle)) / wavelength)
            projection = np.dot(noise_subspace.T.conj(), steering_vector)
            spectrum.append(1 / (np.linalg.norm(projection) ** 2))
        return angles[np.argmax(spectrum)]

    def corr_eigen_ratio_packet(self, csi_packet):
        num_antennas = csi_packet.shape[1]
        if num_antennas < 2:
            return 0
        csi_avg_subcarriers = np.mean(csi_packet, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(csi_avg_subcarriers.T)
            if np.any(np.isnan(corr_matrix)):
                return 0
            eig_vals = np.linalg.eigvalsh(corr_matrix)
            eig_vals = np.sort(eig_vals)[::-1]
            return eig_vals[0] / np.sum(eig_vals)

    def num_peaks_packet(self, csi_packet):
        signal_1d = np.abs(csi_packet[:, 0, 0])
        smoothed = gaussian_filter1d(signal_1d, sigma=1)
        peaks, _ = find_peaks(smoothed, height=np.mean(smoothed))
        return len(peaks)

    def std_amp_packet(self, csi_packet):
        signal_1d = np.abs(csi_packet[:, 0, 0])
        smoothed = gaussian_filter1d(signal_1d, sigma=1)
        return np.std(smoothed)

    def median_amp_packet(self, csi_packet):
        signal_1d = np.abs(csi_packet[:, 0, 0])
        smoothed = gaussian_filter1d(signal_1d, sigma=1)
        return np.median(smoothed)

    def skew_amp_packet(self, csi_packet):
        signal_1d = np.abs(csi_packet[:, 0, 0])
        smoothed = gaussian_filter1d(signal_1d, sigma=1)
        return skew(smoothed)

    def kurtosis_amp_packet(self, csi_packet):
        signal_1d = np.abs(csi_packet[:, 0, 0])
        smoothed = gaussian_filter1d(signal_1d, sigma=1)
        return kurtosis(smoothed)