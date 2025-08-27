"""
feature_detection.py

CSI Feature Detection Module

This module implements feature extraction methods for CSI data analysis based on the paper
"Hands-on Wireless Sensing with Wi-Fi" by Yang et al. The implementation focuses on
extracting meaningful features for human activity recognition and pose detection.

Mathematical Background:
----------------------
1. Time-domain Features:
   - Signal Energy: E = Σ|x[n]|²
     Represents total signal power over time window
   - Signal Variance: σ² = E[(X - μ)²]
     Measures signal spread around mean
   - Zero Crossing Rate: ZCR = (1/T)Σ|sign(x[n]) - sign(x[n-1])|
     Indicates frequency of signal polarity changes
   - Peak Analysis: Local maxima in signal amplitude
     Identifies significant signal events

2. Frequency-domain Features:
   - Power Spectral Density (PSD): S(f) = |FFT(x[n])|²/N
     Shows power distribution across frequencies
   - Spectral Centroid: fc = Σ(f·S(f))/ΣS(f)
     Represents the "center of mass" of the spectrum
   - Dominant Frequencies: argmax(S(f))
     Identifies most significant frequency components

3. Statistical Features:
   - Kurtosis: κ = E[(X-μ)⁴]/σ⁴
     Measures "tailedness" of amplitude distribution
   - Skewness: γ = E[(X-μ)³]/σ³
     Measures asymmetry of amplitude distribution
   - Antenna Correlation: ρij = cov(Xi,Xj)/(σi·σj)
     Measures relationship between antenna pairs

4. Dimensionality Reduction:
   - PCA: X = UΣVᵀ
     Extracts principal components for feature compression

Reference Paper Methodology:
--------------------------
The feature extraction follows these key principles from the paper:
1. Multi-domain analysis for comprehensive feature capture
2. Statistical robustness through proper normalization
3. Dimensionality reduction for efficient processing
4. Motion detection through energy variance analysis

Configuration:
-------------
Feature extraction parameters are configured in:
src/sanitization/config/feature_config.json

Dependencies:
------------
- numpy: Numerical computations
- scipy: Signal processing functions
- sklearn: PCA implementation
- matplotlib: Feature visualization
"""

import os
import json
import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from csiread_lib.atheros_csi_read import Atheros
from pathlib import Path


class CSIFeatureDetector:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature detector.

        Args:
            config_path: Optional path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "config", "feature_config.json"
            )

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.sample_rate = self.config["sampling_rate"]
        self.window_sizes = self.config["window_sizes"]
        self.freq_bands = self.config["frequency_bands"]
        self.pca_config = self.config["pca"]
        self.motion_config = self.config["motion_detection"]

        self._last_features = None
        self._feature_history = []

        # Initialize visualization if needed
        try:
            from .feature_visualization import FeatureVisualizer

            self.visualizer = FeatureVisualizer()
        except ImportError:
            self.visualizer = None

    def extract_time_features(
        self, csi_data: np.ndarray, window_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Extract time-domain features from CSI data.

        Args:
            csi_data: CSI data array of shape (packets, subcarriers, rx_ant, tx_ant)
            window_size: Size of the sliding window for feature extraction

        Returns:
            Dictionary containing different time-domain features
        """
        # Compute signal energy (sum of squared magnitudes)
        energy = np.sum(np.abs(csi_data) ** 2, axis=1)

        # Compute signal variance over time
        variance = np.var(np.abs(csi_data), axis=0)

        # Detect peaks in the signal magnitude
        peaks = self._detect_peaks(np.abs(csi_data))

        # Calculate zero crossing rate
        zcr = self._zero_crossing_rate(csi_data)

        return {
            "energy": energy,
            "variance": variance,
            "peaks": peaks,
            "zero_crossing_rate": zcr,
        }

    def extract_frequency_features(
        self, csi_data: np.ndarray, nperseg: int = 256
    ) -> Dict[str, np.ndarray]:
        """
        Extract frequency-domain features using FFT and spectral analysis.

        Args:
            csi_data: CSI data array
            nperseg: Length of each segment for spectral analysis

        Returns:
            Dictionary containing frequency-domain features
        """
        # Compute FFT
        freqs = fftfreq(csi_data.shape[0], 1 / self.sample_rate)
        fft_data = fft(np.abs(csi_data), axis=0)

        # Compute power spectral density
        freq, psd = signal.welch(
            np.abs(csi_data), fs=self.sample_rate, nperseg=nperseg, axis=0
        )

        # Calculate spectral centroid
        centroid = np.sum(freq[:, np.newaxis] * psd, axis=0) / np.sum(psd, axis=0)

        # Find dominant frequencies
        dominant_freq = freq[np.argmax(psd, axis=0)]

        return {
            "spectral_centroid": centroid,
            "dominant_frequency": dominant_freq,
            "psd": psd,
            "frequencies": freq,
        }

    def extract_statistical_features(
        self, csi_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from CSI data.

        Args:
            csi_data: CSI data array

        Returns:
            Dictionary containing statistical features
        """
        # Calculate amplitude and phase
        amplitude = np.abs(csi_data)
        phase = np.angle(csi_data)

        # Compute kurtosis and skewness
        kurtosis = stats.kurtosis(amplitude, axis=0)
        skewness = stats.skew(amplitude, axis=0)

        # Calculate correlation between antennas
        corr_matrix = np.zeros((csi_data.shape[2], csi_data.shape[2]))
        for i in range(csi_data.shape[2]):
            for j in range(csi_data.shape[2]):
                corr_matrix[i, j] = np.corrcoef(
                    amplitude[:, 0, i, 0].flatten(), amplitude[:, 0, j, 0].flatten()
                )[0, 1]

        return {
            "kurtosis": kurtosis,
            "skewness": skewness,
            "antenna_correlation": corr_matrix,
        }

    def compute_pca_features(
        self, csi_data: np.ndarray, n_components: int = 3
    ) -> np.ndarray:
        """
        Apply PCA to reduce dimensionality and extract principal components.

        Args:
            csi_data: CSI data array
            n_components: Number of principal components to retain

        Returns:
            Array containing the principal components
        """
        # Reshape data for PCA
        X = np.abs(csi_data).reshape(csi_data.shape[0], -1)

        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)

        return components

    def detect_motion(
        self, csi_data: np.ndarray, threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Detect motion events in CSI data using energy-based detection.

        Args:
            csi_data: CSI data array
            threshold: Detection threshold for motion events

        Returns:
            Tuple of (motion_detected, confidence_score)
        """
        # Compute signal energy
        energy = np.sum(np.abs(csi_data) ** 2, axis=(1, 2, 3))

        # Compute energy variance
        energy_var = np.var(energy)

        # Motion detection based on energy variance
        motion_detected = energy_var > threshold
        confidence = energy_var / threshold

        return motion_detected, confidence

    def _detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Helper function to detect peaks in the signal."""
        return signal.argrelextrema(signal, np.greater)[0]

    def _zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calculate zero crossing rate of the signal."""
        return np.mean(np.abs(np.diff(np.signbit(signal.real))))

    def _compute_entropy(self, signal: np.ndarray) -> float:
        """Compute Shannon entropy of the signal."""
        bins = np.histogram(signal, bins="auto")[0]
        probs = bins / len(signal)
        return -np.sum(probs * np.log2(probs + 1e-10))


def load_and_process_csi(file_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load CSI data from .dat file and extract features.

    Args:
        file_path: Path to the CSI data file

    Returns:
        Tuple of (features_dict, metadata)
    """
    # Load CSI data
    reader = Atheros(file_path)
    reader.read()
    csi_data = reader.csi

    # Create feature detector
    detector = CSIFeatureDetector(sample_rate=100.0)  # Adjust sample rate as needed

    # Extract features
    time_features = detector.extract_time_features(csi_data)
    freq_features = detector.extract_frequency_features(csi_data)
    stat_features = detector.extract_statistical_features(csi_data)
    pca_features = detector.compute_pca_features(csi_data)

    # Combine all features
    features = {
        "time_domain": time_features,
        "frequency_domain": freq_features,
        "statistical": stat_features,
        "pca": pca_features,
    }

    # Prepare metadata
    metadata = {
        "timestamp": reader.timestamp,
        "csi_shape": csi_data.shape,
        "num_packets": len(reader.timestamp),
        "sample_rate": 100.0,  # Adjust as needed
    }

    return features, metadata


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Example data file
    data_file = "dataset/standing/root/standing-1-0-1.dat"

    # Load and process CSI data
    features, metadata = load_and_process_csi(data_file)

    # Print feature statistics
    print("CSI Feature Extraction Results:")
    print(f"Number of packets processed: {metadata['num_packets']}")
    print(f"Data shape: {metadata['csi_shape']}")
    print("\nFeature Summary:")
    print("Time-domain features:", features["time_domain"].keys())
    print("Frequency-domain features:", features["frequency_domain"].keys())
    print("Statistical features:", features["statistical"].keys())
    print("PCA components shape:", features["pca"].shape)

    # Visualize features
    from feature_visualization import FeatureVisualizer

    visualizer = FeatureVisualizer()

    # Create output directory for plots
    output_dir = Path("feature_plots")
    output_dir.mkdir(exist_ok=True)

    # Generate and save all plots
    visualizer.save_all_plots(features, str(output_dir), "standing_features")
    print(f"\nPlots saved in: {output_dir}")

    # Show a specific plot interactively
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(np.abs(features["time_domain"]["energy"]))
    plt.title("Signal Energy Over Time")

    plt.subplot(122)
    plt.plot(
        features["frequency_domain"]["frequencies"],
        features["frequency_domain"]["psd"].mean(axis=1),
    )
    plt.title("Average Power Spectral Density")
    plt.tight_layout()
    plt.show()
