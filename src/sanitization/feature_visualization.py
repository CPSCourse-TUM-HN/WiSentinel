"""
feature_visualization.py

Visualization utilities for CSI feature detection.

This module provides functions to visualize various aspects of CSI data and extracted features:
1. Time-domain visualizations:
   - Signal amplitude/phase over time
   - Energy and variance plots
   - Peak detection visualization
2. Frequency-domain visualizations:
   - Power spectral density
   - Spectral centroid
   - Dominant frequency components
3. Statistical visualizations:
   - Correlation matrices
   - PCA components
   - Feature distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
import json
import os
from pathlib import Path


def load_config():
    """Load visualization configuration."""
    config_path = Path(__file__).parent / "config" / "feature_config.json"
    with open(config_path, "r") as f:
        return json.load(f)["visualization"]


class FeatureVisualizer:
    def __init__(self, style: str = "seaborn"):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        self.config = load_config()
        plt.style.use(style)

    def plot_time_domain(
        self,
        features: Dict[str, np.ndarray],
        time_axis: Optional[np.ndarray] = None,
        title: str = "Time Domain Features",
    ) -> plt.Figure:
        """
        Plot time-domain features.

        Args:
            features: Dictionary of time-domain features
            time_axis: Optional time axis values
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(title)

        # Plot signal energy
        ax = axes[0]
        energy = features["energy"]
        if time_axis is None:
            time_axis = np.arange(len(energy))
        ax.plot(time_axis, energy)
        ax.set_title("Signal Energy")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")

        # Plot variance
        ax = axes[1]
        variance = features["variance"]
        ax.plot(variance.flatten())
        ax.set_title("Signal Variance")
        ax.set_xlabel("Subcarrier")
        ax.set_ylabel("Variance")

        # Plot peaks
        ax = axes[2]
        peaks = features["peaks"]
        signal = np.abs(energy)  # Use energy as base signal
        ax.plot(time_axis, signal)
        ax.plot(time_axis[peaks], signal[peaks], "ro", label="Peaks")
        ax.set_title("Peak Detection")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_frequency_domain(
        self, features: Dict[str, np.ndarray], title: str = "Frequency Domain Features"
    ) -> plt.Figure:
        """
        Plot frequency-domain features.

        Args:
            features: Dictionary of frequency-domain features
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(title)

        # Plot power spectral density
        ax = axes[0]
        freq = features["frequencies"]
        psd = features["psd"]
        ax.semilogy(freq, psd.mean(axis=1))
        ax.set_title("Power Spectral Density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power/Frequency")
        ax.grid(True)

        # Plot spectral centroid
        ax = axes[1]
        centroid = features["spectral_centroid"]
        ax.plot(centroid, label="Spectral Centroid")
        ax.axhline(centroid.mean(), color="r", linestyle="--", label="Mean Centroid")
        ax.set_title("Spectral Centroid")
        ax.set_xlabel("Subcarrier")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_statistical_features(
        self, features: Dict[str, np.ndarray], title: str = "Statistical Features"
    ) -> plt.Figure:
        """
        Plot statistical features.

        Args:
            features: Dictionary of statistical features
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title)

        # Plot kurtosis
        ax = axes[0, 0]
        kurtosis = features["kurtosis"]
        ax.bar(np.arange(len(kurtosis.flatten())), kurtosis.flatten())
        ax.set_title("Kurtosis")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Kurtosis Value")

        # Plot skewness
        ax = axes[0, 1]
        skewness = features["skewness"]
        ax.bar(np.arange(len(skewness.flatten())), skewness.flatten())
        ax.set_title("Skewness")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Skewness Value")

        # Plot antenna correlation matrix
        ax = axes[1, 0]
        corr_matrix = features["antenna_correlation"]
        sns.heatmap(corr_matrix, ax=ax, annot=True, cmap="coolwarm")
        ax.set_title("Antenna Correlation Matrix")

        # Keep one subplot empty for potential additional features
        axes[1, 1].axis("off")

        plt.tight_layout()
        return fig

    def plot_pca_components(
        self, pca_features: np.ndarray, title: str = "PCA Components"
    ) -> plt.Figure:
        """
        Plot PCA components.

        Args:
            pca_features: Array of PCA components
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        if pca_features.shape[1] < 2:
            raise ValueError("Need at least 2 PCA components for visualization")

        fig = plt.figure(figsize=(12, 8))

        if pca_features.shape[1] >= 3:
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                pca_features[:, 0],
                pca_features[:, 1],
                pca_features[:, 2],
                c=np.arange(len(pca_features)),
                cmap="viridis",
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
        else:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                pca_features[:, 0],
                pca_features[:, 1],
                c=np.arange(len(pca_features)),
                cmap="viridis",
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        plt.colorbar(scatter, label="Time")
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def save_all_plots(
        self,
        features: Dict[str, Dict[str, np.ndarray]],
        output_dir: str,
        base_filename: str,
    ) -> None:
        """
        Save all feature plots to files.

        Args:
            features: Dictionary containing all features
            output_dir: Directory to save plots
            base_filename: Base name for plot files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Time domain plots
        fig = self.plot_time_domain(features["time_domain"])
        fig.savefig(os.path.join(output_dir, f"{base_filename}_time_domain.png"))
        plt.close(fig)

        # Frequency domain plots
        fig = self.plot_frequency_domain(features["frequency_domain"])
        fig.savefig(os.path.join(output_dir, f"{base_filename}_freq_domain.png"))
        plt.close(fig)

        # Statistical feature plots
        fig = self.plot_statistical_features(features["statistical"])
        fig.savefig(os.path.join(output_dir, f"{base_filename}_statistical.png"))
        plt.close(fig)

        # PCA plots
        fig = self.plot_pca_components(features["pca"])
        fig.savefig(os.path.join(output_dir, f"{base_filename}_pca.png"))
        plt.close(fig)
