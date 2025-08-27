import os
import sys
import json
import numpy as np
import click
import time
from typing import Optional
from collections import deque
from data_transfer import data_sending, data_receiving
from csiread_lib.atheros_csi_read import Atheros, visualize_csi
from sanitization.sanitize_denoising import RealTimeSanitizeDenoising, load_csi_data

# Add csi_streaming_realtime to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
csi_streaming_path = os.path.abspath(
    os.path.join(current_dir, "..", "..", "..", "csi_streaming_realtime")
)
if csi_streaming_path not in sys.path:
    sys.path.insert(0, csi_streaming_path)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def load_config(config_name):
    # Try to find config in several possible locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "config", config_name),
        os.path.join(os.path.dirname(__file__), "..", "config", config_name),
        os.path.join(
            os.path.dirname(__file__), "..", "sanitization", "config", config_name
        ),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    # If no config found, use defaults
    if config_name == "sanitize_config.json":
        return {
            "nonlinear": True,
            "sto": True,
            "rco": True,
            "cfo": False,
            "default_calib_path": "data_files/calibration/no_person.dat",
        }
    elif config_name == "live_config.json":
        return {
            "max_stored_packets": 1000,
            "default_receiver_host": "localhost",
            "default_listen_port": 8000,
            "default_output_path": "data_files/live_output",
            "default_max_packets": None,
            "default_timeout": None,
            "visualize_data": True,
        }

    raise FileNotFoundError(f"Could not find config file: {config_name}")


sanitize_config = load_config("sanitize_config.json")
live_config = load_config("live_config.json")

# Constants
DEFAULT_RX_NUM = 3
DEFAULT_TX_NUM = 2
DEFAULT_TONES = 56
DEFAULT_PAYLOAD_LENGTH = 0
DEFAULT_CALIB_PATH = "data_files/calibration/no_person.dat"


def load_calibration_data(file_path: str) -> np.ndarray:
    """
    Load calibration data from either .dat or .npy file.

    Args:
        file_path: Path to the calibration file (.dat or .npy)

    Returns:
        np.ndarray: The loaded calibration CSI data
    """
    if file_path.endswith(".npy"):
        return np.load(file_path), None
    elif file_path.endswith(".dat"):
        return load_csi_data(file_path)
    else:
        raise ValueError("Calibration file must be either .dat or .npy format")


class CSILiveProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.processed_packets = deque(maxlen=live_config["max_stored_packets"])
        self.sanitized_data = []
        self.sanitizer = None

    def choose_dat_file(self, data_dir: Optional[str] = None) -> Optional[str]:
        if data_dir is None:
            data_dir = os.path.join(self.base_path, "..", "..", "data_files")

        dat_files = [f for f in os.listdir(data_dir) if f.endswith(".dat")]

        if not dat_files:
            print(f"No .dat files found in {data_dir}")
            return None

        print("Available .dat files:")
        for i, file in enumerate(dat_files):
            print(f"{i + 1}. {file}")

        choice = click.prompt("Select a file number", type=int, default=1)

        if 1 <= choice <= len(dat_files):
            return os.path.join(data_dir, dat_files[choice - 1])
        else:
            print("Invalid choice. Using the first file.")
            return os.path.join(data_dir, dat_files[0])

    def process_live_csi_data(
        self,
        mode: str,
        sanitize: bool = False,
        calib_path: Optional[str] = None,
        output_path: Optional[str] = None,
        receiver_host: Optional[str] = None,
        max_packets: Optional[int] = None,
        timeout: Optional[float] = None,
        nonlinear: Optional[bool] = None,
        sto: Optional[bool] = None,
        rco: Optional[bool] = None,
        cfo: Optional[bool] = None,
    ) -> None:
        if sanitize:
            if calib_path is None:
                raise ValueError("Calibration file is required for sanitization.")
            self.sanitizer = RealTimeSanitizeDenoising(calib_path)

        if mode == "send":
            infile = self.choose_dat_file()
            if not infile:
                print("No valid .dat file selected. Exiting.")
                return
            data_sending.run_client(
                receiver_host or live_config["default_receiver_host"],
                receiver_port=live_config["default_listen_port"],
                infile=infile,
                mode="stream",
            )
        elif mode == "receive":
            if not output_path:
                output_path = live_config["default_output_path"]

            # Initialize feature detector
            feature_detector = RealTimeFeatureDetector(
                buffer_size=1000,
                feature_window=100,
                update_interval=0.1,
                enable_visualization=True,
            )

            def process_packet(packet):
                # First apply sanitization if enabled
                if self.sanitizer:
                    sanitized_packet = self.sanitizer.process_packet(
                        packet,
                        nonlinear=(
                            nonlinear
                            if nonlinear is not None
                            else sanitize_config["nonlinear"]
                        ),
                        sto=sto if sto is not None else sanitize_config["sto"],
                        rco=rco if rco is not None else sanitize_config["rco"],
                        cfo=cfo if cfo is not None else sanitize_config["cfo"],
                    )
                else:
                    sanitized_packet = packet

                # Process features in real-time
                update_needed, features = feature_detector.process_packet(
                    sanitized_packet, time.time()
                )

                # Update visualization if needed
                if update_needed and features:
                    feature_detector.plot_live_features(features)

                    # Log motion events
                    if features["motion"]["detected"]:
                        click.echo(
                            f"Motion detected! Confidence: {features['motion']['confidence']:.2f}"
                        )
                    self.processed_packets.append(sanitized_packet)
                    if output_path:
                        np.save(
                            os.path.join(
                                output_path,
                                f"sanitized_packet_{self.sanitizer.packet_count}.npy",
                            ),
                            sanitized_packet,
                        )
                else:
                    self.processed_packets.append(packet)
                    if output_path:
                        np.save(
                            os.path.join(
                                output_path,
                                f"raw_packet_{len(self.processed_packets)}.npy",
                            ),
                            packet,
                        )
                print(f"Processed packet {len(self.processed_packets)}")

            data_receiving.run_server(
                listen_port=live_config["default_listen_port"],
                outfile_path=output_path,
                mode="stream",
                max_packets=max_packets or live_config["default_max_packets"],
                timeout=timeout or live_config["default_timeout"],
                packet_callback=process_packet,
            )

            if live_config["visualize_data"] and self.processed_packets:
                all_csi = np.array(list(self.processed_packets))
                csidata = Atheros(
                    output_path, ntxnum=3
                )  # Explicitly set ntxnum=3 to handle files with more transmit antennas
                csidata.csi = all_csi
                csidata.read()  # Initialize other properties
                visualize_csi(csidata, title="Processed CSI Data")


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["send", "receive"]),
    required=True,
    help="Operation mode: send or receive",
)
@click.option(
    "--sanitize/--no-sanitize", default=True, help="Enable sanitization of CSI data"
)
@click.option("--calib-file", "-c", help="Path to calibration file (for sanitization)")
@click.option("--output", "-o", help="Output file for received data")
@click.option("--receiver-host", help="Receiver IP address (for send mode)")
@click.option("--max-packets", type=int, help="Maximum number of packets to receive")
@click.option("--timeout", type=float, help="Timeout in seconds for receiving")
@click.option(
    "--nonlinear/--no-nonlinear",
    default=sanitize_config["nonlinear"],
    help="Apply nonlinear sanitization",
)
@click.option(
    "--detect-features/--no-features",
    default=False,
    help="Enable real-time feature detection",
)
@click.option(
    "--detect-motion/--no-motion",
    default=False,
    help="Enable real-time motion detection",
)
@click.option(
    "--visualize/--no-visualize",
    default=False,
    help="Enable real-time feature visualization",
)
@click.option(
    "--sto/--no-sto", default=sanitize_config["sto"], help="Apply STO correction"
)
@click.option(
    "--rco/--no-rco", default=sanitize_config["rco"], help="Apply RCO correction"
)
@click.option(
    "--cfo/--no-cfo", default=sanitize_config["cfo"], help="Apply CFO correction"
)
def main(
    mode: str,
    sanitize: bool,
    calib_file: Optional[str],
    output: Optional[str],
    receiver_host: Optional[str],
    max_packets: Optional[int],
    timeout: Optional[float],
    nonlinear: bool,
    sto: bool,
    rco: bool,
    cfo: bool,
) -> None:
    """
    Process live CSI data with real-time sanitization and visualization.

    This command-line interface provides two main modes:
    1. Send mode: Stream CSI data from a .dat file to a receiver
    2. Receive mode: Listen for incoming CSI data and process it in real-time

    The receiver mode supports real-time sanitization using the RealTimeSanitizeDenoising
    class, which maintains a buffer of recent packets for smooth processing. Sanitized
    packets can be saved individually and visualized in real-time.

    Sanitization options:
    - Nonlinear amplitude & phase correction
    - Sample Time Offset (STO) calibration
    - Radio Chain Offset (RCO) calibration
    - Carrier Frequency Offset (CFO) estimation

    Usage Examples:
        # Sending mode - will prompt for file selection
        python cli_live.py --mode send --receiver-host 192.168.1.10

        # Receiving mode with sanitization and visualization
        python cli_live.py --mode receive --sanitize --calib-file calibration.dat --output received_data

        # Receiving mode with packet limits
        python cli_live.py --mode receive --max-packets 1000 --timeout 60

        # Full receiving mode with all sanitization options
        python cli_live.py --mode receive --sanitize --calib-file calibration.dat \
            --output received_data --nonlinear --sto --rco --no-cfo
    """
    try:
        processor = CSILiveProcessor(os.path.dirname(os.path.abspath(__file__)))
        processor.process_live_csi_data(
            mode,
            sanitize,
            calib_file,
            output,
            receiver_host,
            max_packets,
            timeout,
            nonlinear,
            sto,
            rco,
            cfo,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
