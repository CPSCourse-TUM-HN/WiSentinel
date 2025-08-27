"""
realtime_features.py

Real-time feature detection for live CSI data streams.
Optimized for low-latency processing and live visualization.

This module extends the feature detection capabilities to work with
streaming CSI data, providing:
1. Buffered processing for smooth feature extraction
2. Real-time visualization updates
3. Motion event detection
4. Live feature statistics
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from collections import deque
from threading import Lock
import time

from .feature_detection import CSIFeatureDetector
from .feature_visualization import FeatureVisualizer


class RealTimeFeatureDetector:
    def __init__(
        self,
        buffer_size: int = 1000,
        feature_window: int = 100,
        update_interval: float = 0.1,
        enable_visualization: bool = True,
    ):
        """
        Initialize real-time feature detector.

        Args:
            buffer_size: Maximum number of packets to keep in buffer
            feature_window: Window size for feature computation
            update_interval: Minimum time between updates (seconds)
            enable_visualization: Whether to enable live plotting
        """
        self.buffer = deque(maxlen=buffer_size)
        self.feature_window = feature_window
        self.update_interval = update_interval
        self.last_update = 0

        # Initialize base feature detector
        self.detector = CSIFeatureDetector()

        # Initialize visualization if enabled
        self.visualizer = FeatureVisualizer() if enable_visualization else None

        # Thread safety
        self.buffer_lock = Lock()

        # Feature history
        self.feature_history = {
            "time": deque(maxlen=buffer_size),
            "freq": deque(maxlen=buffer_size),
            "stats": deque(maxlen=buffer_size),
        }

        # Motion detection state
        self.motion_state = {"detected": False, "confidence": 0.0, "last_event": None}

    def process_packet(
        self, csi_packet: np.ndarray, timestamp: float
    ) -> Tuple[bool, Dict[str, np.ndarray]]:
        """
        Process a single CSI packet in real-time.

        Args:
            csi_packet: Single CSI data packet
            timestamp: Packet timestamp

        Returns:
            Tuple of (update_needed, features)
        """
        with self.buffer_lock:
            self.buffer.append((csi_packet, timestamp))

        # Check if we should compute new features
        current_time = time.time()
        update_needed = (current_time - self.last_update) >= self.update_interval

        if update_needed and len(self.buffer) >= self.feature_window:
            features = self._compute_current_features()
            self.last_update = current_time
            return True, features

        return False, None

    def _compute_current_features(self) -> Dict[str, np.ndarray]:
        """Compute features from current buffer window."""
        with self.buffer_lock:
            # Get the most recent window of packets
            recent_packets = list(self.buffer)[-self.feature_window :]
            packets, timestamps = zip(*recent_packets)
            csi_data = np.stack(packets)

        # Extract features
        time_features = self.detector.extract_time_features(csi_data)
        freq_features = self.detector.extract_frequency_features(csi_data)
        stat_features = self.detector.extract_statistical_features(csi_data)

        # Update feature history
        self.feature_history["time"].append(time_features)
        self.feature_history["freq"].append(freq_features)
        self.feature_history["stats"].append(stat_features)

        # Check for motion
        motion_detected, confidence = self.detector.detect_motion(csi_data)
        self.motion_state.update(
            {
                "detected": motion_detected,
                "confidence": confidence,
                "last_event": (
                    time.time() if motion_detected else self.motion_state["last_event"]
                ),
            }
        )

        return {
            "time_domain": time_features,
            "frequency_domain": freq_features,
            "statistical": stat_features,
            "motion": self.motion_state,
            "timestamp": timestamps[-1],
        }

    def get_feature_statistics(self) -> Dict[str, np.ndarray]:
        """Compute statistics over the feature history."""
        if not self.feature_history["time"]:
            return None

        # Compute mean and std of key features
        energy_history = np.array(
            [f["energy"].mean() for f in self.feature_history["time"]]
        )
        freq_history = np.array(
            [f["spectral_centroid"].mean() for f in self.feature_history["freq"]]
        )

        return {
            "energy_mean": np.mean(energy_history),
            "energy_std": np.std(energy_history),
            "frequency_mean": np.mean(freq_history),
            "frequency_std": np.std(freq_history),
            "motion_events": sum(
                1
                for f in self.feature_history["time"]
                if f.get("motion", {}).get("detected", False)
            ),
        }

    def plot_live_features(self, features: Dict[str, np.ndarray]) -> None:
        """Update live visualization with new features."""
        if not self.visualizer:
            return

        # Create/update plots
        self.visualizer.plot_time_domain(
            features["time_domain"], title="Live Time Domain Features"
        )

        if features["motion"]["detected"]:
            plt.suptitle(
                f"Motion Detected! (Confidence: {features['motion']['confidence']:.2f})"
            )

        plt.pause(0.01)  # Allow plot to update
