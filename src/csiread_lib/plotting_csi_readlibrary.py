import socket
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import time
from csiread import Atheros
from scipy import signal

# ---------------- Enhanced Configuration ------------------
PORT = 5500
HOST = ""
HEADER_LEN = 0
SAMPLING_FREQ = 15
WINDOW_SECONDS = 30
DISPLAY_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
GAIN = 10.0
MOTION_THRESHOLD = 1.5  # STD threshold for motion detection
SMOOTHING_WINDOW = 5  # Moving average window
NRX = 3  # Number of RX antennas
NTX = 3  # Number of TX antennas


# ---------------- Enhanced Utility ------------------
def nearest_pow_2(x):
    return int(math.pow(2, math.ceil(np.log2(x)) if x > 0 else 1))


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# ---------------- Enhanced Packet Receiver ------------------
class PacketReceiver:
    def __init__(self, port, host="", header_len=0):
        self.header_len = header_len
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.1)  # Non-blocking with timeout
        print(f"[INFO] Listening for UDP packets on port {port}...")

    def receive(self):
        try:
            data, _ = self.sock.recvfrom(4096)
            return data[self.header_len :]
        except socket.timeout:
            return None


# ---------------- Enhanced Signal Extractor ------------------
class SignalExtractor:
    def __init__(self):
        self.parser = Atheros(None, nrxnum=NRX, ntxnum=NTX, if_report=False)
        self.prev_phase = None
        self.bad_packet_count = 0

    def extract_features(self, payload: bytes):
        # Validate input
        if not payload or len(payload) < 100:  # Minimum expected CSI packet size
            self.bad_packet_count += 1
            if self.bad_packet_count % 50 == 0:  # Don't spam on continuous errors
                print(
                    f"[WARN] Received {self.bad_packet_count} bad packets (empty/short)"
                )
            return None, None, None

        try:
            # Create a memoryview to avoid copies
            with memoryview(payload) as mv:
                if not mv.tobytes():  # Additional validation
                    return None, None, None

                # Use csiread's built-in buffer interface
                self.parser.read(mv)

                if not hasattr(self.parser, "csi") or len(self.parser.csi) == 0:
                    return None, None, None

                csi = self.parser.csi[-1]  # Latest frame

                # Amplitude features
                amplitudes = np.abs(csi)
                mean_amp = np.mean(amplitudes)

                # Phase processing with safety checks
                phases = np.angle(csi)
                phase_diff = np.zeros_like(phases)
                if self.prev_phase is not None:
                    try:
                        phase_diff = self._unwrap_phase(phases - self.prev_phase)
                    except ValueError as ve:
                        print(f"[WARN] Phase unwrapping error: {str(ve)}")
                        phase_diff = phases - self.prev_phase  # Use raw difference
                self.prev_phase = phases

                # Subcarrier correlation with fallback
                try:
                    corr = self._calculate_subcarrier_correlation(amplitudes)
                except:
                    corr = 0.0

                return mean_amp, np.var(phase_diff), corr

        except Exception as e:
            print(f"[ERROR] CSI parsing failed: {str(e)}")
            if hasattr(e, "errno"):
                print(f"Socket errno: {e.errno}")
            return None, None, None

    def _unwrap_phase(self, phase_diff):
        """Handle phase wrapping between -π and π with checks"""
        phase_diff = np.asarray(phase_diff)
        if phase_diff.size == 0:
            return phase_diff
        return np.unwrap(phase_diff, axis=0)

    def _calculate_subcarrier_correlation(self, amplitudes):
        """Calculate correlation between adjacent subcarriers with validation"""
        flattened = amplitudes.reshape(amplitudes.shape[0], -1)
        if flattened.shape[1] < 2:
            return 0.0
        corr_matrix = np.corrcoef(flattened.T)
        return corr_matrix[0, 1] if corr_matrix.size >= 4 else 0.0


# ---------------- Enhanced Plotter with Motion Detection ------------------
class Plotter:
    def __init__(self, sampling_freq, window_seconds, gain):
        self.sampling_freq = sampling_freq
        self.display_samples = sampling_freq * window_seconds
        self.gain = gain

        # Create figure with 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            nrows=3, figsize=(15, 12), gridspec_kw={"height_ratios": [1, 1, 1]}
        )
        plt.subplots_adjust(hspace=0.5)
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

        # FFT parameters
        self.wlen = 1
        self.per_lap = 0.9
        self.mult = 8.0
        self.nfft_base = nearest_pow_2(self.wlen * self.sampling_freq)
        self.nlap_base = int(self.nfft_base * self.per_lap)
        self.mult = nearest_pow_2(self.mult) * self.nfft_base

        # Motion detection history
        self.motion_history = []
        self.motion_threshold = MOTION_THRESHOLD

    def update(self, stream, phase_vars, corrs, fps):
        if len(stream) < 2:  # Need at least 2 samples
            return

        # Smooth the stream
        smoothed_stream = moving_average(stream, SMOOTHING_WINDOW)

        # Plot 1: CSI Amplitude and Motion Detection
        self.ax1.clear()
        self.ax1.plot(smoothed_stream, "b-", label="CSI Amplitude")

        # Motion detection logic
        std_dev = np.std(smoothed_stream)
        mean_val = np.mean(smoothed_stream)
        motion_mask = smoothed_stream > (mean_val + self.motion_threshold * std_dev)

        # Mark motion events
        motion_indices = np.where(motion_mask)[0]
        if len(motion_indices) > 0:
            self.ax1.plot(
                motion_indices,
                smoothed_stream[motion_indices],
                "ro",
                label="Motion Detected",
            )

        self.ax1.axhline(
            mean_val + self.motion_threshold * std_dev,
            color="r",
            linestyle="--",
            label="Threshold",
        )
        self.ax1.set_title(f"CSI Amplitude (Motion Detection) - FPS: {fps:.1f}")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.legend()

        # Plot 2: Phase Variance
        self.ax2.clear()
        if len(phase_vars) == len(smoothed_stream):
            self.ax2.plot(phase_vars, "g-")
            self.ax2.set_title("Phase Variance Over Time")
            self.ax2.set_ylabel("Phase Variance")

        # Plot 3: Spectrogram
        self.ax3.clear()
        try:
            nfft = min(self.nfft_base, len(smoothed_stream))
            nlap = min(self.nlap_base, nfft - 1)

            spec, freqs, bins, im = self.ax3.specgram(
                smoothed_stream,
                NFFT=nfft,
                Fs=self.sampling_freq,
                noverlap=nlap,
                cmap="viridis",
            )
            self.ax3.set_title("Spectrogram")
            self.ax3.set_xlabel("Time (packets)")
            self.ax3.set_ylabel("Frequency (Hz)")
        except Exception as e:
            print(f"[WARN] Spectrogram error: {str(e)}")

        self.fig.canvas.draw()
        plt.pause(0.001)


# ---------------- Enhanced Main ------------------
def main():
    receiver = PacketReceiver(PORT, HOST, HEADER_LEN)
    extractor = SignalExtractor()
    plotter = Plotter(SAMPLING_FREQ, WINDOW_SECONDS, GAIN)

    stream = []
    phase_vars = []
    corrs = []
    last_time = datetime.now()
    fps = 0

    while True:
        try:
            payload = receiver.receive()
            if payload is None:
                continue

            mean_amp, phase_diff, corr = extractor.extract_features(payload)

            if mean_amp is None:
                continue

            stream.append(mean_amp)
            phase_vars.append(np.var(phase_diff) if phase_diff is not None else 0)
            corrs.append(corr if corr is not None else 0)

            # Maintain fixed window size
            if len(stream) > DISPLAY_SAMPLES:
                stream = stream[-DISPLAY_SAMPLES:]
                phase_vars = phase_vars[-DISPLAY_SAMPLES:]
                corrs = corrs[-DISPLAY_SAMPLES:]

            # Calculate FPS
            now = datetime.now()
            fps = 0.9 * fps + 0.1 * (
                1.0 / max(0.001, (now - last_time).total_seconds())
            )
            last_time = now

            # Update plots
            plotter.update(stream, phase_vars, corrs, fps)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
            break
        except Exception as e:
            print(f"[ERROR] Main loop: {str(e)}")
            continue


if __name__ == "__main__":
    main()
