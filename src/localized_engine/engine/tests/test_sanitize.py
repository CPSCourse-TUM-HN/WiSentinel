import unittest
import numpy as np
from data_processing.sanitize_denoising import set_template, sanitize_csi
from scipy.io import loadmat
import os


class TestCSISanitization(unittest.TestCase):
    def setUp(self):
        # Compute project root relative to this test file
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        # Paths to test files
        self.calib_mat = "/Users/shivamsingh/PycharmProjects/WiSentinel/data_files/calibration/csi_calib_test.mat"
        self.src_mat = "/data_files/csi_src_test.mat"
        # Ensure files exist
        self.assertTrue(
            os.path.isfile(self.calib_mat),
            f"Calibration file not found: {self.calib_mat}",
        )
        self.assertTrue(
            os.path.isfile(self.src_mat), f"Source CSI file not found: {self.src_mat}"
        )

    def test_set_template_output(self):
        # Load calibration CSI
        mat = loadmat(self.calib_mat)
        calib_csi = next(mat[k] for k in mat if not k.startswith("__"))
        # Wrap to 4D if needed (add packet dim)
        if calib_csi.ndim == 3:
            calib_csi = calib_csi[None, ...]
        T, S, A, L = calib_csi.shape
        linear_interval = np.arange(1, S + 1)
        tpl = set_template(calib_csi, linear_interval)
        expected_shape = (1, S, A, L)
        self.assertIsInstance(tpl, np.ndarray)
        self.assertEqual(
            tpl.shape,
            expected_shape,
            f"Template shape mismatch: {tpl.shape} vs {expected_shape}",
        )

    def test_sanitize_preserves_shape_and_no_nan(self):
        # Load source CSI from MAT file
        mat_src = loadmat(self.src_mat)
        csi_src = next(mat_src[k] for k in mat_src if not k.startswith("__"))
        # Wrap to 4D
        if csi_src.ndim == 3:
            csi_src = csi_src[None, ...]
        P, S, A, L = csi_src.shape
        linear_interval = np.arange(1, S + 1)
        # Load calibration CSI
        mat_calib = loadmat(self.calib_mat)
        calib_csi = next(mat_calib[k] for k in mat_calib if not k.startswith("__"))
        if calib_csi.ndim == 3:
            calib_csi = calib_csi[None, ...]
        tpl = set_template(calib_csi, linear_interval)
        # Sanitize
        csi_out = sanitize_csi(csi_src.copy(), tpl, linear_interval)
        # Assertions
        self.assertEqual(csi_out.shape, csi_src.shape, "Sanitized CSI shape changed")
        self.assertFalse(np.isnan(csi_out).any(), "Sanitized CSI contains NaNs")


if __name__ == "__main__":
    unittest.main()
