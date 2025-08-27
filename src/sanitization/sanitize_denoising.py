"""
csi_sanitization.py

CSI Sanitization Module (Phase 3)

Provides functions to apply a sequence of calibrations and denoising steps on raw CSI arrays:
  - Template creation (amplitude & phase) from calibration CSI
  - Nonlinear amplitude & phase correction
  - Sampling Time Offset (STO) calibration
  - Carrier Frequency Offset (CFO) estimation
  - Radio Chain Offset (RCO) calibration

Workflow:
  1. Generate a template using `set_template(calib_csi, linear_interval)`
     from a calibration CSI dataset.
  2. Call `sanitize_csi(csi, template, linear_interval)` to apply all steps and return
     a sanitized CSI array.

Functions:
  - set_template(calib_csi: np.ndarray, linear_interval: np.ndarray) -> np.ndarray
  - apply_nonlinear_template(csi: np.ndarray, template: np.ndarray) -> np.ndarray
  - sto_calib_mul(csi: np.ndarray) -> np.ndarray
  - estimate_cfo(csi: np.ndarray) -> float
  - rco_calib(csi: np.ndarray) -> np.ndarray
  - apply_rco(csi: np.ndarray, rco_vec: np.ndarray) -> np.ndarray
  - sanitize_csi(csi: np.ndarray, template: np.ndarray, linear_interval: np.ndarray) -> np.ndarray

CSI array shape: (packets, subcarriers, rx_antennas, tx_antennas)
"""

import numpy as np

from typing import Optional, Tuple
from csiread_lib.atheros_csi_read import Atheros
import os
import json
import logging


class SanitizeDenoising:
    """
    CSI sanitization pipeline:
    1. Template creation (external, e.g. set_template)
    2. Nonlinearity fix (apply_nonlinear_template)
    3. STO correction (remove_sto)
    4. RCO correction (remove_rco)
    5. CFO correction (apply_cfo_correction)
    6. Sanitize call (sanitize_csi_batch)
    """

    # 1. Template creation
    def set_template(
        self, calib_csi: np.ndarray, linear_interval: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        MATLAB-style set_template: Generate a CSI calibration template for amplitude and phase.
        - Amplitude: Normalize per-packet, then average across packets.
        - Phase: Remove linear trend (fit) in the linear interval, average nonlinear error, set template phase to zero in linear region.
        Robust to NaN/Inf in calibration data.
        Args:
            calib_csi: Calibration CSI, shape [T S A L]
            linear_interval: Indices of linear phase region (1D array)
        Returns:
            csi_calib_template: Complex template, shape [1 S A L]
        """
        if linear_interval is None:
            linear_interval = np.arange(*self.config["linear_interval"])
        # Ensure linear_interval is within bounds
        max_index = calib_csi.shape[1] - 1
        linear_interval = linear_interval[linear_interval <= max_index]
        T, S, A, L = calib_csi.shape
        # Clean calibration data: replace NaN/Inf with zeros (or could use nanmean/nanstd)
        calib_csi_clean = np.where(np.isfinite(calib_csi), calib_csi, 0)
        # Amplitude normalization per packet (MATLAB: csi_amp ./ mean(csi_amp,2)), robust
        csi_amp = np.abs(calib_csi_clean)  # [T S A L]
        mean_amp = np.nanmean(
            np.where(np.isfinite(csi_amp), csi_amp, np.nan), axis=1, keepdims=True
        )  # [T 1 A L]
        mean_amp = np.where(np.isfinite(mean_amp), mean_amp, 1e-8)
        csi_amp_norm = csi_amp / (mean_amp + 1e-8)  # [T S A L]
        csi_amp_template = np.nanmean(
            np.where(np.isfinite(csi_amp_norm), csi_amp_norm, np.nan),
            axis=0,
            keepdims=True,
        )  # [1 S A L]
        csi_amp_template = np.where(
            np.isfinite(csi_amp_template), csi_amp_template, 1.0
        )
        # Phase: unwrap along subcarriers (MATLAB: unwrap(angle(csi_calib),[],2)), robust
        csi_phase = np.unwrap(np.angle(calib_csi_clean), axis=1)  # [T S A L]
        nonlinear_phase_error = np.zeros_like(csi_phase)
        for p in range(T):
            for a in range(A):
                for e in range(L):
                    y = csi_phase[p, linear_interval, a, e]
                    x = linear_interval
                    # Linear fit (MATLAB: fit(...,'poly1'))
                    if len(x) < 2 or np.any(~np.isfinite(y)):
                        linear_model = np.array([0, 0])
                    else:
                        linear_model = np.polyfit(x, y, 1)
                    fit_line = np.polyval(linear_model, np.arange(S))
                    nonlinear_phase_error[p, :, a, e] = csi_phase[p, :, a, e] - fit_line
        # Average nonlinear phase error across packets, robust
        csi_phase_template = np.nanmean(
            np.where(np.isfinite(nonlinear_phase_error), nonlinear_phase_error, np.nan),
            axis=0,
            keepdims=True,
        )  # [1 S A L]
        csi_phase_template = np.where(
            np.isfinite(csi_phase_template), csi_phase_template, 0.0
        )
        # Set template phase to zero on linear_interval (MATLAB: csi_phase_template(1,linear_interval,:,:) = 0)
        for idx in linear_interval:
            csi_phase_template[0, idx, :, :] = 0
        # Combine amplitude and phase into complex template
        csi_calib_template = csi_amp_template * np.exp(
            1j * csi_phase_template
        )  # [1 S A L]
        # Clean template: replace any NaN/Inf with safe defaults
        csi_calib_template = np.where(
            np.isfinite(csi_calib_template), csi_calib_template, 1.0
        )
        logging.info(
            f"Template created. Shape: {csi_calib_template.shape}. Any NaN/Inf: {not np.all(np.isfinite(csi_calib_template))}"
        )
        return csi_calib_template

    # 2. Nonlinearity fix
    def apply_nonlinear_template(
        self, csi: np.ndarray, template: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        result = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"apply_nonlinear_template: Packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
                continue
            # MATLAB: amp = abs(pkt), phase = unwrap(angle(pkt),[],2)
            amp = np.abs(pkt)
            phase = np.unwrap(np.angle(pkt), axis=0)
            # Divide by template amplitude, keep phase (MATLAB: ./ template)
            res_pkt = amp * np.exp(1j * phase) / (np.abs(template[0]) + eps)
            if not np.all(np.isfinite(res_pkt)):
                logging.warning(
                    f"apply_nonlinear_template: Output for packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
            else:
                result[i] = res_pkt
        return result

    # 3. STO correction
    def remove_sto(self, csi: np.ndarray) -> np.ndarray:
        # Packet-level NaN/Inf check and handling, robust to shape mismatches
        result = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"remove_sto: Packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
                continue
            phase = np.angle(pkt)
            # Unwrap along subcarrier axis (assume axis=0 for [S, ...])
            unwrapped_phase = np.unwrap(phase, axis=0)
            S = pkt.shape[0]
            rest_shape = pkt.shape[1:]
            # Compute slope for each [A, L] pair
            if S < 2:
                logging.error(
                    f"remove_sto: Not enough subcarriers (S={S}) for packet {i}, zeroing out."
                )
                result[i] = 0
                continue
            # slope shape: [A, L]
            slope = (unwrapped_phase[-1, ...] - unwrapped_phase[0, ...]) / (S - 1)
            # correction shape: [S, A, L] (broadcast slope over S)
            correction = np.arange(S)[:, None, None] * slope[None, ...]
            # corrected_phase shape: [S, A, L]
            corrected_phase = phase - correction
            try:
                res_pkt = np.abs(pkt) * np.exp(1j * corrected_phase)
            except Exception as e:
                logging.error(
                    f"remove_sto: Broadcasting error for packet {i}: {e}. Shapes: phase {phase.shape}, correction {correction.shape}"
                )
                result[i] = 0
                continue
            if not np.all(np.isfinite(res_pkt)):
                logging.warning(
                    f"remove_sto: Output for packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
            else:
                result[i] = res_pkt
        return result

    def sto_calib_mul(self, csi: np.ndarray) -> np.ndarray:
        # Packet-level NaN/Inf check and handling
        result = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"sto_calib_mul: Packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
                continue
            ref = pkt[pkt.shape[0] // 2, :, :]
            res_pkt = pkt * np.conj(ref[None, :, :])
            if not np.all(np.isfinite(res_pkt)):
                logging.warning(
                    f"sto_calib_mul: Output for packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
            else:
                result[i] = res_pkt
        return result

    # 4. RCO correction
    def remove_rco(self, csi: np.ndarray) -> np.ndarray:
        # Defensive: initialize running_state if needed (original behavior)
        if self.running_state is None:
            self.running_state = {
                "phase_offset": np.zeros(csi.shape[1:]),
                "amplitude_offset": np.zeros(csi.shape[1:]),
            }
        result = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"remove_rco: Packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
                continue
            phase_diff = np.angle(pkt) - self.running_state["phase_offset"]
            amplitude_ratio = np.abs(pkt) / (
                self.running_state["amplitude_offset"] + 1e-8
            )
            res_pkt = amplitude_ratio * np.exp(1j * phase_diff)
            if not np.all(np.isfinite(res_pkt)):
                logging.warning(
                    f"remove_rco: Output for packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
            else:
                result[i] = res_pkt
        return result

    def rco_calib(self, csi_calib: np.ndarray) -> np.ndarray:
        """
        MATLAB-style RCO estimation, with packet-level NaN/Inf check.
        Args:
            csi_calib: Calibration CSI, shape [T, S, A, L]
        Returns:
            est_rco: Estimated RCO, shape [A]
        """
        if not np.all(np.isfinite(csi_calib)):
            logging.warning(
                "rco_calib: Calibration data contains NaN/Inf, returning zeros."
            )
            return np.zeros(csi_calib.shape[2])
        antenna_num = csi_calib.shape[2]
        csi_phase = np.unwrap(np.angle(csi_calib), axis=0)  # unwrap along time
        avg_phase = np.zeros(antenna_num)
        for a in range(antenna_num):
            avg_phase[a] = np.mean(csi_phase[:, :, a, 0])  # use first HT-LTF
        est_rco = avg_phase - avg_phase[0]
        if not np.all(np.isfinite(est_rco)):
            logging.warning("rco_calib: Output contains NaN/Inf, returning zeros.")
            return np.zeros_like(est_rco)
        return est_rco

    def apply_rco(self, csi: np.ndarray, rco_vec: np.ndarray) -> np.ndarray:
        # Packet-level NaN/Inf check and handling
        result = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(f"apply_rco: Packet {i} contains NaN/Inf, zeroing out.")
                result[i] = 0
                continue
            res_pkt = pkt * np.exp(-1j * rco_vec[None, :, None])
            if not np.all(np.isfinite(res_pkt)):
                logging.warning(
                    f"apply_rco: Output for packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
            else:
                result[i] = res_pkt
        return result

    # 5. CFO correction
    def estimate_cfo_matlab(
        self, csi_src: np.ndarray, delta_time: float = 4e-6
    ) -> float:
        """
        MATLAB-style CFO estimation using two HT-LTFs (see provided MATLAB code).
        Args:
            csi_src: CSI data with at least two HT-LTFs, shape [T S A L]
            delta_time: Time interval between HT-LTFs (default 4e-6 seconds)
        Returns:
            est_cfo: Estimated frequency offset (float)
        """
        if csi_src.shape[-1] < 2:
            raise ValueError("CSI data must have at least two HT-LTFs (last dimension)")
        phase_1 = np.angle(csi_src[..., 0])  # [T S A]
        phase_2 = np.angle(csi_src[..., 1])  # [T S A]
        phase_diff = phase_2 - phase_1  # [T S A]
        # Average over antennas (axis=-1 if [T S], else axis=2)
        if phase_diff.ndim == 3:
            phase_diff_mean = np.mean(phase_diff, axis=2)  # [T S]
        else:
            phase_diff_mean = phase_diff  # fallback
        # Average over subcarriers (axis=1 if [T S], else axis=0)
        if phase_diff_mean.ndim == 2:
            phase_diff_mean = np.mean(phase_diff_mean, axis=1)  # [T]
        est_cfo = np.mean(phase_diff_mean / delta_time)  # scalar
        return est_cfo

    def apply_cfo_correction(self, csi: np.ndarray, cfo_slope: float) -> np.ndarray:
        """
        Apply Carrier Frequency Offset (CFO) correction to CSI data (MATLAB-style).
        Args:
            csi: Input CSI, shape [N S A L]
            cfo_slope: Estimated CFO slope (float)
        Returns:
            CFO-corrected CSI, same shape as input
        """
        N = csi.shape[0]
        correction = np.exp(-1j * cfo_slope * np.arange(N))[:, None, None, None]
        return csi * correction

    # 6. Sanitization calls
    def sanitize_csi_batch(
        self,
        csi: np.ndarray,
        template: Optional[np.ndarray] = None,
        nonlinear: Optional[bool] = None,
        sto: Optional[bool] = None,
        rco: Optional[bool] = None,
        cfo: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Apply all sanitization steps to a batch of CSI packets in the following order:
        1. Nonlinearity fix (using template)
        2. STO correction
        3. RCO correction
        4. CFO correction
        Args:
            csi: Input CSI, shape [N S A L]
            template: Sanitization template (created externally)
            nonlinear, sto, rco, cfo: Which steps to apply
        Returns:
            Sanitized CSI, shape [N S A L]
        """
        nonlinear = nonlinear if nonlinear is not None else self.config["nonlinear"]
        sto = sto if sto is not None else self.config["sto"]
        rco = rco if rco is not None else self.config["rco"]
        cfo = cfo if cfo is not None else self.config["cfo"]
        sanitized = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"sanitize_csi_batch: Packet {i} contains NaN/Inf, zeroing out."
                )
                sanitized[i] = 0
                continue
            pkt_sanitized = pkt.copy()
            # 1. Nonlinearity fix
            if nonlinear and template is not None:
                pkt_sanitized = self.apply_nonlinear_template(
                    pkt_sanitized[None, ...], template
                )[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_batch: Packet {i} NaN/Inf after nonlinear correction, zeroing out."
                    )
                    sanitized[i] = 0
                    continue
            # 2. STO correction
            if sto:
                pkt_sanitized = self.remove_sto(pkt_sanitized[None, ...])[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_batch: Packet {i} NaN/Inf after STO correction, zeroing out."
                    )
                    sanitized[i] = 0
                    continue
            # 3. RCO correction
            if rco:
                pkt_sanitized = self.remove_rco(pkt_sanitized[None, ...])[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_batch: Packet {i} NaN/Inf after RCO correction, zeroing out."
                    )
                    sanitized[i] = 0
                    continue
            # 4. CFO correction
            if cfo:
                try:
                    cfo_slope = self.estimate_cfo_matlab(pkt_sanitized[None, ...])
                except Exception as e:
                    logging.error(
                        f"MATLAB-style CFO estimation failed for packet {i}: {e}"
                    )
                    sanitized[i] = 0
                    continue
                pkt_sanitized = self.apply_cfo_correction(
                    pkt_sanitized[None, ...], cfo_slope
                )[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_batch: Packet {i} NaN/Inf after CFO correction, zeroing out."
                    )
                    sanitized[i] = 0
                    continue
            sanitized[i] = pkt_sanitized
        return sanitized

    def sanitize_csi_selective(
        self,
        csi: np.ndarray,
        template: Optional[np.ndarray] = None,
        nonlinear: Optional[bool] = None,
        sto: Optional[bool] = None,
        rco: Optional[bool] = None,
        cfo: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Selectively apply sanitization techniques to CSI data, with packet-level NaN/Inf handling.

        Args:
            csi: Input CSI data
            template: Sanitization template
            nonlinear: Apply nonlinear sanitization
            sto: Apply STO (Sample Time Offset) correction
            rco: Apply RCO (Residual Carrier Offset) correction
            cfo: Apply CFO (Carrier Frequency Offset) correction

        Returns:
            Sanitized CSI data
        """
        nonlinear = nonlinear if nonlinear is not None else self.config["nonlinear"]
        sto = sto if sto is not None else self.config["sto"]
        rco = rco if rco is not None else self.config["rco"]
        cfo = cfo if cfo is not None else self.config["cfo"]
        valid_packets = []
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"sanitize_csi_selective: Packet {i} contains NaN/Inf, dropping packet."
                )
                continue
            pkt_sanitized = pkt.copy()
            if nonlinear and template is not None:
                pkt_sanitized = self.apply_nonlinear_template(
                    pkt_sanitized[None, ...], template
                )[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_selective: Packet {i} NaN/Inf after nonlinear correction, dropping packet."
                    )
                    continue
            if sto:
                pkt_sanitized = self.remove_sto(pkt_sanitized[None, ...])[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_selective: Packet {i} NaN/Inf after STO correction, dropping packet."
                    )
                    continue
            if rco:
                pkt_sanitized = self.remove_rco(pkt_sanitized[None, ...])[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_selective: Packet {i} NaN/Inf after RCO correction, dropping packet."
                    )
                    continue
            if cfo:
                try:
                    cfo_slope = self.estimate_cfo_matlab(pkt_sanitized[None, ...])
                except Exception as e:
                    logging.error(
                        f"MATLAB-style CFO estimation failed for packet {i}: {e}"
                    )
                    continue
                pkt_sanitized *= np.exp(
                    -1j
                    * cfo_slope
                    * np.arange(pkt_sanitized.shape[0])[:, None, None, None]
                )[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_csi_selective: Packet {i} NaN/Inf after CFO correction, dropping packet."
                    )
                    continue
            valid_packets.append(pkt_sanitized)
        if valid_packets:
            return np.stack(valid_packets, axis=0)
        else:
            return np.empty((0,) + csi.shape[1:], dtype=csi.dtype)

    def sanitize_packet(
        self,
        csi_packet: np.ndarray,
        nonlinear: bool = True,
        sto: bool = True,
        rco: bool = True,
        cfo: bool = False,
    ) -> np.ndarray:
        """
        Apply all sanitization steps to a single CSI packet (or batch of packets),
        closely following MATLAB pipeline logic:
        - Nonlinear amplitude/phase correction (template-based)
        - STO removal (detrend phase across subcarriers)
        - RCO removal (subtract running reference phase/amplitude)
        - CFO correction (optional, not always used in MATLAB pipeline)
        Each step is robust to NaN/Inf and operates per-packet.
        Args:
            csi_packet: Input CSI, shape [N S A L] or [S A L] (N=packets)
            nonlinear, sto, rco, cfo: Which steps to apply
        Returns:
            Sanitized CSI, shape [N S A L]
        """
        if self.template is None:
            raise ValueError("Template not set. Call set_template() first.")

        self.packet_count += 1

        # Reshape the packet if necessary (MATLAB: always 4D)
        if csi_packet.ndim == 2:
            csi_packet = csi_packet.reshape(1, *csi_packet.shape)

        # Initialize running state if it's the first packet
        if self.running_state is None:
            self.running_state = {
                "phase_offset": np.zeros(csi_packet.shape[1:]),
                "amplitude_offset": np.zeros(csi_packet.shape[1:]),
            }

        sanitized_packet = np.empty_like(csi_packet)
        for i in range(csi_packet.shape[0]):
            pkt = csi_packet[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"sanitize_packet: Packet {i} contains NaN/Inf, zeroing out."
                )
                sanitized_packet[i] = 0
                continue
            pkt_sanitized = pkt.copy()
            # Nonlinear amplitude/phase correction (template-based, MATLAB: set_template/apply)
            if nonlinear and self.template is not None:
                pkt_sanitized = self.apply_nonlinear_template(
                    pkt_sanitized[None, ...], self.template
                )[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_packet: Packet {i} NaN/Inf after nonlinear correction, zeroing out."
                    )
                    sanitized_packet[i] = 0
                    continue
            # STO removal (MATLAB: remove linear phase slope across subcarriers)
            if sto:
                pkt_sanitized = self.remove_sto(pkt_sanitized[None, ...])[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_packet: Packet {i} NaN/Inf after STO correction, zeroing out."
                    )
                    sanitized_packet[i] = 0
                    continue
            # RCO removal (MATLAB: subtract reference phase/amplitude)
            if rco:
                pkt_sanitized = self.remove_rco(pkt_sanitized[None, ...])[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_packet: Packet {i} NaN/Inf after RCO correction, zeroing out."
                    )
                    sanitized_packet[i] = 0
                    continue
            # CFO correction (MATLAB: optional, not always used)
            if cfo:
                cfo_slope = self.estimate_cfo(pkt_sanitized[None, ...])
                pkt_sanitized *= np.exp(
                    -1j
                    * cfo_slope
                    * np.arange(pkt_sanitized.shape[0])[:, None, None, None]
                )[0]
                if not np.all(np.isfinite(pkt_sanitized)):
                    logging.warning(
                        f"sanitize_packet: Packet {i} NaN/Inf after CFO correction, zeroing out."
                    )
                    sanitized_packet[i] = 0
                    continue
            sanitized_packet[i] = pkt_sanitized
        self.update_running_state(sanitized_packet)
        return sanitized_packet

    # 7. Utility and helper functions
    def update_running_state(self, packet: np.ndarray):
        """
        Update the running state (reference phase/amplitude) for RCO correction.
        - Only update if all values are finite.
        - Running state is an exponential moving average (MATLAB: similar to recursive mean).
        Args:
            packet: Most recent sanitized packet(s), shape [N S A L]
        """
        if not self.running_state is None:
            # Check for contamination in running state
            if not (
                np.all(np.isfinite(self.running_state["phase_offset"]))
                and np.all(np.isfinite(self.running_state["amplitude_offset"]))
            ):
                logging.error(
                    "Running state contaminated with NaN/Inf. Resetting running state."
                )
                self.running_state["phase_offset"] = np.zeros_like(
                    self.running_state["phase_offset"]
                )
                self.running_state["amplitude_offset"] = np.zeros_like(
                    self.running_state["amplitude_offset"]
                )
        if np.all(np.isfinite(packet)):
            alpha = 1 / self.packet_count
            # MATLAB: running mean of phase/amplitude for reference
            self.running_state["phase_offset"] = (1 - alpha) * self.running_state[
                "phase_offset"
            ] + alpha * np.angle(packet[0])
            self.running_state["amplitude_offset"] = (1 - alpha) * self.running_state[
                "amplitude_offset"
            ] + alpha * np.abs(packet[0])
        else:
            logging.warning("Skipped running state update due to NaN/Inf in packet")

    def normalize_amplitude(
        self, csi: np.ndarray, axis: int = 1, eps: float = 1e-8
    ) -> np.ndarray:
        """
        Normalize amplitude along specified axis (MATLAB: csi_amp ./ mean(csi_amp,axis)), robust to NaN/Inf.
        Args:
            csi: Input CSI, shape [...]
            axis: Axis along which to normalize
            eps: Small value to avoid division by zero
        Returns:
            Amplitude-normalized CSI (same shape)
        """
        amp = np.abs(csi)
        # Use nanmean to ignore NaN/Inf in mean calculation
        mean_amp = np.nanmean(
            np.where(np.isfinite(amp), amp, np.nan), axis=axis, keepdims=True
        )
        mean_amp = np.where(
            np.isfinite(mean_amp), mean_amp, eps
        )  # fallback for all-NaN
        return csi / (mean_amp + eps)

    def remove_linear_phase(self, csi: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Remove linear phase slope along the specified axis (MATLAB: detrend phase).
        Args:
            csi: Input CSI, shape [...]
            axis: Axis along which to remove linear phase (default: subcarriers)
        Returns:
            CSI with linear phase removed (same shape as input)
        """
        result = np.empty_like(csi)
        for i in range(csi.shape[0]):
            pkt = csi[i]
            if not np.all(np.isfinite(pkt)):
                logging.warning(
                    f"remove_linear_phase: Packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
                continue
            phase = np.angle(pkt)  # Extract phase of each element in the packet
            unwrapped_phase = np.unwrap(
                phase, axis=axis
            )  # Unwrap phase along the specified axis to avoid discontinuities
            # Linear fit along axis
            x = np.arange(pkt.shape[axis])  # Create an array of indices for the axis
            # Collapse all other axes for fit
            shape = list(pkt.shape)
            shape.pop(axis)  # Remove the axis being processed
            for idx in np.ndindex(
                *shape
            ):  # Iterate over all combinations of the remaining axes
                slicer = list(idx)
                slicer.insert(
                    axis, slice(None)
                )  # Insert a slice for the axis being processed
                y = unwrapped_phase[
                    tuple(slicer)
                ]  # Select the phase values for this slice
                if len(x) < 2:
                    linear_model = np.array(
                        [0, 0]
                    )  # Not enough points for a fit, use zero slope/intercept
                else:
                    linear_model = np.polyfit(
                        x, y, 1
                    )  # Fit a line (degree 1 polynomial) to the phase values
                fit_line = np.polyval(
                    linear_model, x
                )  # Evaluate the fitted line at each index
                corrected = (
                    phase[tuple(slicer)] - fit_line
                )  # Subtract the fitted line from the original phase
                # Assign back
                unwrapped_phase[tuple(slicer)] = (
                    corrected  # Store the corrected phase back in the array
                )
            res_pkt = np.abs(pkt) * np.exp(
                1j * unwrapped_phase
            )  # Reconstruct the packet with corrected phase and original amplitude
            if not np.all(np.isfinite(res_pkt)):
                logging.warning(
                    f"remove_linear_phase: Output for packet {i} contains NaN/Inf, zeroing out."
                )
                result[i] = 0
            else:
                result[i] = res_pkt
        return result

    def detrend_phase(self, phase: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Remove linear trend from phase along specified axis (MATLAB: detrend).
        Args:
            phase: Phase array, shape [...]
            axis: Axis along which to detrend
        Returns:
            Detrended phase array (same shape)
        """
        # Can be understood as isolating the signal fluctuations.
        result = np.empty_like(phase)
        x = np.arange(phase.shape[axis])
        shape = list(phase.shape)
        shape.pop(axis)
        for idx in np.ndindex(*shape):
            slicer = list(idx)
            slicer.insert(axis, slice(None))
            y = phase[tuple(slicer)]
            if len(x) < 2:
                linear_model = np.array([0, 0])
            else:
                linear_model = np.polyfit(x, y, 1)
            fit_line = np.polyval(linear_model, x)
            result[tuple(slicer)] = y - fit_line
        return result

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "sanitize_config.json"
        )
        with open(config_path, "r") as f:
            return json.load(f)

    def __init__(self, calib_path: Optional[str] = None):
        self.config = self._load_config()
        if calib_path:
            self.calib_path = os.path.abspath(calib_path)
        else:
            # Use the project root as the base for the relative path
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            self.calib_path = os.path.join(
                project_root, self.config["default_calib_path"]
            )

        if not os.path.exists(self.calib_path):
            raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")

        self.template = None
        self.running_state = None
        self.packet_count = 0


class RealTimeSanitizeDenoising(SanitizeDenoising):
    """
    Real-time CSI sanitization class for processing streaming CSI packets.
    Maintains running state and buffer for online denoising and calibration.
    """

    def __init__(self, calib_path: Optional[str] = None, buffer_size: int = 100):
        super().__init__(calib_path)
        self.buffer = []
        self.buffer_size = buffer_size

    def process_packet(
        self,
        csi_packet: np.ndarray,
        nonlinear: bool = True,
        sto: bool = True,
        rco: bool = True,
        cfo: bool = False,
    ) -> np.ndarray:
        """
        Process a single CSI packet in real time, applying sanitization steps.
        Maintains a buffer of recent sanitized packets for smoothing if needed.
        """
        sanitized = self.sanitize_packet(
            csi_packet, nonlinear=nonlinear, sto=sto, rco=rco, cfo=cfo
        )
        self.buffer.append(sanitized)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return sanitized

    def get_buffer_average(self) -> Optional[np.ndarray]:
        """
        Return the average of the sanitized packets in the buffer (for smoothing).
        """
        if not self.buffer:
            return None
        return np.mean(self.buffer, axis=0)

    def reset_buffer(self):
        self.buffer = []


def load_csi_data(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load CSI data from a .dat or .npy file.

    Args:
        file_path: Path to the file (.dat or .npy format)

    Returns:
        Tuple containing CSI data array and metadata dictionary.
        For .npy files, metadata will be None.
    """
    if file_path.endswith(".npy"):
        return np.load(file_path), None
    elif file_path.endswith(".dat"):
        reader = Atheros(file_path)
        reader.read()

        csi_data = reader.csi
        metadata = {
            "timestamp": reader.timestamp,
            "csi_len": reader.csi_len,
            "tx_channel": reader.tx_channel,
            "err_info": reader.err_info,
            "noise_floor": reader.noise_floor,
            "Rate": reader.Rate,
            "bandWidth": reader.bandWidth,
            "num_tones": reader.num_tones,
            "nr": reader.nr,
            "nc": reader.nc,
            "rssi": reader.rssi,
            "rssi_1": reader.rssi_1,
            "rssi_2": reader.rssi_2,
            "rssi_3": reader.rssi_3,
            "payload_len": reader.payload_len,
            "payload": reader.payload,
        }

    return csi_data, metadata


# Example usage
if __name__ == "__main__":
    calib_path = "path/to/calibration.dat"
    data_path = "path/to/csi_data.dat"

    import argparse

    parser = argparse.ArgumentParser(description="CSI Sanitization CLI")
    parser.add_argument(
        "--calib", type=str, required=True, help="Path to calibration CSI file"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to CSI data file")
    parser.add_argument(
        "--method",
        type=str,
        default="selective",
        choices=["selective", "batch"],
        help="Sanitization method",
    )
    parser.add_argument(
        "--nonlinear", action="store_true", help="Apply nonlinear correction"
    )
    parser.add_argument("--sto", action="store_true", help="Apply STO correction")
    parser.add_argument("--rco", action="store_true", help="Apply RCO correction")
    parser.add_argument("--cfo", action="store_true", help="Apply CFO correction")
    parser.add_argument(
        "--out", type=str, default=None, help="Output .npy file for sanitized CSI"
    )
    parser.add_argument(
        "--template_linear_start",
        type=int,
        default=20,
        help="Start index for template linear interval",
    )
    parser.add_argument(
        "--template_linear_end",
        type=int,
        default=39,
        help="End index for template linear interval (exclusive)",
    )
    args = parser.parse_args()

    # Load calibration data
    calib_csi, _ = load_csi_data(args.calib)
    sanitizer = SanitizeDenoising(args.calib)
    linear_interval = np.arange(args.template_linear_start, args.template_linear_end)
    template = sanitizer.set_template(calib_csi, linear_interval)

    # Load CSI data
    csi_data, metadata = load_csi_data(args.data)

    # Choose sanitization method
    if args.method == "batch":
        sanitized_csi = sanitizer.sanitize_csi_batch(
            csi_data,
            template,
            nonlinear=args.nonlinear,
            sto=args.sto,
            rco=args.rco,
            cfo=args.cfo,
        )
    else:
        sanitized_csi = sanitizer.sanitize_csi_selective(
            csi_data,
            template,
            nonlinear=args.nonlinear,
            sto=args.sto,
            rco=args.rco,
            cfo=args.cfo,
        )

    print("Sanitization complete.")
    print(f"Original CSI shape: {csi_data.shape}")
    print(f"Sanitized CSI shape: {sanitized_csi.shape}")
    if args.out:
        np.save(args.out, sanitized_csi)
        print(f"Sanitized CSI saved to {args.out}")

