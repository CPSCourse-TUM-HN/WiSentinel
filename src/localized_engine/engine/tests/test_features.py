import unittest
import numpy as np
from scipy.signal import stft
from Localized_Engine.Trash.features.simple_amplitude import simple_amplitude
from Localized_Engine.Trash.features.naive_tof import naive_tof
from Localized_Engine.Trash.features.advanced_features import (
    naive_spectrum,
    naive_aoa,
    delay_spread,
    spatial_correlation,
)


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # Synthetic CSI: 4 packets, 8 subcarriers, 2 RX antennas, 1 TX chain
        self.P, self.S, self.A, self.L = 4, 8, 2, 1
        # Random complex CSI
        np.random.seed(0)
        self.csi = np.random.randn(
            self.P, self.S, self.A, self.L
        ) + 1j * np.random.randn(self.P, self.S, self.A, self.L)
        self.bw = 20e6  # bandwidth for ToF and delay
        self.fs = 100  # sampling rate for STFT
        self.antenna_distances = [0.05]  # distance between 2 antennas
        self.fc = 5e9  # center frequency for AoA

    def test_simple_amplitude(self):
        X = simple_amplitude(self.csi)
        # Should be shape (P, S*A*L)
        self.assertEqual(X.shape, (self.P, self.S * self.A * self.L))
        # Values should equal abs(csi).flatten
        expected = np.abs(self.csi).reshape(self.P, -1)
        self.assertTrue(np.allclose(X, expected))

    def test_naive_tof_shape_and_nonneg(self):
        tof = naive_tof(self.csi, self.bw)
        # Shape (P, A)
        self.assertEqual(tof.shape, (self.P, self.A))
        # All values non-negative
        self.assertTrue((tof >= 0).all())

    def test_naive_spectrum_shape(self):
        seq = np.abs(self.csi).mean(axis=(1, 2, 3))
        # Compute reference STFT
        f, t, Z = stft(seq, fs=self.fs, nperseg=4, noverlap=2)
        spec = naive_spectrum(self.csi, self.fs, nperseg=4, noverlap=2)
        # Shapes must match
        self.assertEqual(spec.shape, Z.shape)
        # Magnitudes should match
        self.assertTrue(np.allclose(spec, np.abs(Z)))

    def test_naive_aoa_zero_phase(self):
        # CSI with zero phase difference
        csi_const = np.ones((self.P, self.S, self.A, self.L), dtype=complex)
        aoa = naive_aoa(csi_const, self.antenna_distances, self.fc)
        # Shape (P, A-1)
        self.assertEqual(aoa.shape, (self.P, self.A - 1))
        # All zeros
        self.assertTrue(np.allclose(aoa, 0))

    def test_delay_spread_nonneg(self):
        ds = delay_spread(self.csi, self.bw)
        # Shape (P, 2)
        self.assertEqual(ds.shape, (self.P, 2))
        # Mean delay and RMS spread non-negative
        self.assertTrue((ds >= 0).all())

    def test_spatial_correlation_bounds(self):
        sc = spatial_correlation(self.csi)
        # Shape (P, A*(A-1)/2)
        expected_len = self.A * (self.A - 1) // 2
        self.assertEqual(sc.shape, (self.P, expected_len))
        # Correlations in [-1,1]
        self.assertTrue((sc >= -1).all() and (sc <= 1).all())


if __name__ == "__main__":
    unittest.main()
