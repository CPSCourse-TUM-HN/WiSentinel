import os
import sys
import numpy as np
import click
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

# Add the src directory to the Python path
src_dir = str(Path(__file__).resolve().parent.parent)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from sanitization.sanitize_denoising import SanitizeDenoising, load_csi_data
from sanitization.feature_detection import CSIFeatureDetector
from sanitization.feature_visualization import FeatureVisualizer


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


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Default CSI hardware configuration parameters
DEFAULT_RX_NUM = 3  # Number of receiving antennas
DEFAULT_TX_NUM = 3  # Number of transmitting antennas
DEFAULT_TONES = 56  # Number of OFDM subcarriers (frequency bins)
DEFAULT_PAYLOAD_LENGTH = 10  # Expected payload length in bytes
DEFAULT_OUTPUT = "sanitized_csi.dat"  # Default output filename

# Path to calibration data for CSI sanitization
DEFAULT_CALIB_PATH = os.path.join("data_files", "calibration", "no_person.dat")

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class CSIFileError(Exception):
    """
    Custom exception for CSI file handling errors.
    """

    pass


def setup_logging(enable_logging: bool = False) -> Optional[str]:
    """
    Set up logging configuration with rotating file handler and detailed formatting.

    Args:
        enable_logging: Whether to enable logging to file

    Returns:
        Optional[str]: Path to the log file if logging is enabled, None otherwise
    """
    if not enable_logging:
        return None

    # Use logs directory from repository root
    repo_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Create log file with timestamp and descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"csi_processing_{timestamp}.log"

    # Detailed log format
    log_format = (
        "%(asctime)s [%(levelname)8s] "
        "%(filename)s:%(lineno)d - "
        "%(funcName)s(): %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", date_format)
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Log file created at {log_file}")
    return str(log_file)


# ============================================================================
# MAIN CSI PROCESSING CLASS
# ============================================================================


class CSIProcessor:
    def process_csi_data(
        self,
        data_dir: str,
        sanitize: bool,
        calib_file: Optional[str],
        output: Optional[str],
        save_sanitized: bool,
        nonlinear: bool = True,
        sto: bool = True,
        rco: bool = True,
        cfo: bool = False,
        extract_features: bool = False,
        detect_motion: bool = False,
        visualize: bool = False,
        method: str = "selective",
        template_linear_start: int = 20,
        template_linear_end: int = 39,
    ) -> None:
        if sanitize:
            if calib_file is None:
                raise ValueError("Calibration file is required for sanitization.")

            calib_file_path = os.path.abspath(calib_file)
            if not os.path.exists(calib_file_path):
                # Try relative to the current working directory
                cwd_calib_path = os.path.join(os.getcwd(), calib_file)
                if os.path.exists(cwd_calib_path):
                    calib_file_path = cwd_calib_path
                else:
                    logging.error(f"Calibration file not found: {calib_file_path}")
                    raise FileNotFoundError(
                        f"Calibration file not found: {calib_file_path}"
                    )

            logging.info(f"Using calibration file: {calib_file_path}")
            self.sanitizer = SanitizeDenoising(calib_path=calib_file_path)
            try:
                calib_csi, _ = load_calibration_data(calib_file_path)
                logging.info(f"Calibration CSI shape: {calib_csi.shape}")

                # Check calibration data for NaN values
                calib_nan_count = np.isnan(calib_csi).sum()
                if calib_nan_count > 0:
                    total_elements = calib_csi.size
                    nan_percentage = (calib_nan_count / total_elements) * 100
                    logging.error(
                        f"Calibration file contains {calib_nan_count}/{total_elements} NaN values ({nan_percentage:.2f}%)"
                    )

                    # Analyze NaN distribution in calibration data
                    nan_locations = np.where(np.isnan(calib_csi))
                    calib_nan_by_dimension = {
                        "packets": len(np.unique(nan_locations[0])),
                        "subcarriers": len(np.unique(nan_locations[1])),
                        "rx_antennas": len(np.unique(nan_locations[2])),
                        "tx_antennas": len(np.unique(nan_locations[3])),
                    }

                    logging.error("NaN distribution in calibration file:")
                    logging.error(
                        f"  Packets affected: {calib_nan_by_dimension['packets']}/{calib_csi.shape[0]}"
                    )
                    logging.error(
                        f"  Unique subcarriers affected: {calib_nan_by_dimension['subcarriers']}/{calib_csi.shape[1]}"
                    )
                    logging.error(
                        f"  Unique RX antennas affected: {calib_nan_by_dimension['rx_antennas']}/{calib_csi.shape[2]}"
                    )
                    logging.error(
                        f"  Unique TX antennas affected: {calib_nan_by_dimension['tx_antennas']}/{calib_csi.shape[3]}"
                    )

                    raise ValueError(
                        "Calibration file contains NaN values. Please use a valid calibration file."
                    )

                logging.info(f"Sanitizer config: {self.sanitizer.config}")
                linear_interval = np.arange(template_linear_start, template_linear_end)
                template = self.sanitizer.set_template(calib_csi, linear_interval)
                logging.info(f"Template shape: {template.shape}")

            except Exception as e:
                logging.error(f"Error loading calibration data: {str(e)}")
                raise

        processed_files = 0
        skipped_files = 0

        # Prompt user for file order
        csi_files = [f for f in os.listdir(data_dir) if f.endswith((".dat", ".npy"))]
        if not csi_files:
            print("No .dat or .npy files found in the directory.")
            return
        print("Found the following CSI files:")
        for i, fname in enumerate(csi_files):
            print(f"  [{i}] {fname}")
        order_input = input(
            "Enter a comma-separated list of indices for the order to process files (leave blank for default order): "
        )
        if order_input.strip():
            try:
                indices = [int(idx.strip()) for idx in order_input.split(",")]
                csi_files = [csi_files[i] for i in indices if 0 <= i < len(csi_files)]
            except Exception as e:
                print(f"Invalid input, using default order. Error: {e}")
        else:
            csi_files.sort()

        for filename in csi_files:
            file_path = os.path.join(data_dir, filename)
            try:
                logging.info(f"Processing file: {filename}")
                csi_data, metadata = load_csi_data(file_path)

                if sanitize:
                    # Log pre-sanitization state
                    logging.info(f"Pre-sanitization CSI shape: {csi_data.shape}")
                    initial_nan_mask = np.isnan(csi_data)
                    if np.any(initial_nan_mask):
                        initial_nan_count = np.sum(initial_nan_mask)
                        total_elements = csi_data.size
                        logging.warning(
                            f"File contains {initial_nan_count}/{total_elements} NaN values ({(initial_nan_count/total_elements)*100:.2f}%) before sanitization"
                        )

                    # Apply sanitization
                    # Create sanitized CSI data
                    logging.info(
                        f"Applying sanitization (nonlinear={nonlinear}, sto={sto}, rco={rco}, cfo={cfo})"
                    )

                    # Log NaN statistics before sanitization
                    pre_nan_count = np.isnan(csi_data).sum()
                    pre_total = csi_data.size
                    pre_nan_percentage = (pre_nan_count / pre_total) * 100
                    logging.info(
                        f"Pre-sanitization NaN statistics: {pre_nan_count} NaN values out of {pre_total} total ({pre_nan_percentage:.2f}%)"
                    )

                    # Log locations of NaN values before sanitization
                    if pre_nan_count > 0:
                        pre_nan_locations = np.where(np.isnan(csi_data))
                        nan_by_dimension = {
                            "packets": len(np.unique(pre_nan_locations[0])),
                            "subcarriers": len(np.unique(pre_nan_locations[1])),
                            "rx_antennas": len(np.unique(pre_nan_locations[2])),
                            "tx_antennas": len(np.unique(pre_nan_locations[3])),
                        }
                        logging.info(
                            "Pre-sanitization NaN distribution across dimensions:"
                        )
                        logging.info(
                            f"  Packets affected: {nan_by_dimension['packets']}/{len(csi_data)}"
                        )
                        logging.info(
                            f"  Unique subcarriers affected: {nan_by_dimension['subcarriers']}/{csi_data.shape[1]}"
                        )
                        logging.info(
                            f"  Unique RX antennas affected: {nan_by_dimension['rx_antennas']}/{csi_data.shape[2]}"
                        )
                        logging.info(
                            f"  Unique TX antennas affected: {nan_by_dimension['tx_antennas']}/{csi_data.shape[3]}"
                        )

                    if method == "batch":
                        sanitized_csi = self.sanitizer.sanitize_csi_batch(
                            csi_data,
                            template,
                            nonlinear=nonlinear,
                            sto=sto,
                            rco=rco,
                            cfo=cfo,
                        )
                    else:
                        sanitized_csi = self.sanitizer.sanitize_csi_selective(
                            csi_data,
                            template,
                            nonlinear=nonlinear,
                            sto=sto,
                            rco=rco,
                            cfo=cfo,
                        )
                    logging.info(f"Post-sanitization CSI shape: {sanitized_csi.shape}")

                    # Handle NaN values after sanitization
                    # Get indices of packets that don't contain any NaN values
                    valid_packet_mask = ~np.any(np.isnan(sanitized_csi), axis=(1, 2, 3))
                    invalid_count = len(sanitized_csi) - np.sum(valid_packet_mask)

                    # Calculate NaN percentage
                    nan_percentage = (invalid_count / len(sanitized_csi)) * 100

                    # Warning threshold for high NaN percentages
                    if nan_percentage > 50:
                        logging.error(
                            f"Critical: {nan_percentage:.2f}% of packets affected by NaN values"
                        )
                        logging.error(
                            f"{invalid_count} out of {len(sanitized_csi)} packets have NaN values"
                        )

                        # If more than 90% of packets are affected, suggest possible causes
                        if nan_percentage > 90:
                            logging.error("Possible causes of high NaN rate:")
                            logging.error("1. Calibration file mismatch or corruption")
                            logging.error("2. Input data format inconsistency")
                            logging.error("3. Hardware configuration mismatch")
                            logging.error(
                                "4. Signal quality issues during data collection"
                            )
                            logging.error(
                                "Consider using a different calibration file or reviewing data collection setup."
                            )

                            # Throw an error if all packets are invalid
                            if nan_percentage == 100:
                                raise ValueError(
                                    f"All packets in {filename} contain NaN values after sanitization"
                                )

                    if not np.all(valid_packet_mask):
                        orig_packets = len(sanitized_csi)

                        # Analyze NaN distribution before removing packets
                        nan_locations = np.where(np.isnan(sanitized_csi))
                        nan_by_dimension = {
                            "packets": len(np.unique(nan_locations[0])),
                            "subcarriers": len(np.unique(nan_locations[1])),
                            "rx_antennas": len(np.unique(nan_locations[2])),
                            "tx_antennas": len(np.unique(nan_locations[3])),
                        }

                        # Calculate percentage of NaN values in each dimension
                        nan_percentages = {
                            "packets": (
                                nan_by_dimension["packets"] / sanitized_csi.shape[0]
                            )
                            * 100,
                            "subcarriers": (
                                nan_by_dimension["subcarriers"] / sanitized_csi.shape[1]
                            )
                            * 100,
                            "rx_antennas": (
                                nan_by_dimension["rx_antennas"] / sanitized_csi.shape[2]
                            )
                            * 100,
                            "tx_antennas": (
                                nan_by_dimension["tx_antennas"] / sanitized_csi.shape[3]
                            )
                            * 100,
                        }

                        # Find most affected dimension
                        most_affected = max(nan_percentages.items(), key=lambda x: x[1])
                        logging.warning(
                            f"Most affected dimension: {most_affected[0]} ({most_affected[1]:.2f}% affected)"
                        )

                        # Detailed analysis of packet corruption patterns
                        if nan_by_dimension["packets"] > 0:
                            packet_nan_counts = np.sum(
                                np.isnan(sanitized_csi), axis=(1, 2, 3)
                            )
                            avg_nans_per_corrupt_packet = np.mean(
                                packet_nan_counts[packet_nan_counts > 0]
                            )
                            max_nans_in_packet = np.max(packet_nan_counts)
                            logging.info(f"Corrupt packet statistics:")
                            logging.info(
                                f"  Average NaNs per corrupt packet: {avg_nans_per_corrupt_packet:.2f}"
                            )
                            logging.info(
                                f"  Maximum NaNs in a single packet: {max_nans_in_packet}"
                            )

                        logging.info("NaN distribution across dimensions:")
                        logging.info(
                            f"  Packets affected: {nan_by_dimension['packets']}/{orig_packets}"
                        )
                        logging.info(
                            f"  Unique subcarriers affected: {nan_by_dimension['subcarriers']}/{sanitized_csi.shape[1]}"
                        )
                        logging.info(
                            f"  Unique RX antennas affected: {nan_by_dimension['rx_antennas']}/{sanitized_csi.shape[2]}"
                        )
                        logging.info(
                            f"  Unique TX antennas affected: {nan_by_dimension['tx_antennas']}/{sanitized_csi.shape[3]}"
                        )

                        # Remove packets with NaN values
                        sanitized_csi = sanitized_csi[valid_packet_mask]
                        dropped_packets = orig_packets - len(sanitized_csi)
                        valid_percentage = (len(sanitized_csi) / orig_packets) * 100

                        logging.info(
                            f"Dropped {dropped_packets} packets containing NaN values"
                        )
                        logging.info(
                            f"Retained {len(sanitized_csi)} valid packets ({valid_percentage:.2f}% of original)"
                        )

                        if len(sanitized_csi) == 0:
                            raise ValueError(
                                "All packets contained NaN values after sanitization"
                            )

                    # Log post-sanitization NaN statistics
                    post_nan_count = np.isnan(sanitized_csi).sum()
                    post_total = sanitized_csi.size
                    post_nan_percentage = (
                        (post_nan_count / post_total) * 100 if post_total > 0 else 0
                    )
                    logging.info(
                        f"Post-sanitization NaN statistics: {post_nan_count} NaN values out of {post_total} total ({post_nan_percentage:.2f}%)"
                    )

                    if save_sanitized:
                        self.save_sanitized_data(
                            data_dir, filename, sanitized_csi, calib_file_path
                        )
                    csi_data = sanitized_csi

                # Feature extraction if enabled
                if extract_features or detect_motion or visualize:
                    detector = CSIFeatureDetector()
                    if visualize:
                        visualizer = FeatureVisualizer()

                    # Get base name for feature files
                    feature_base_name = os.path.splitext(filename)[0]

                    # Extract features
                    if extract_features or visualize:
                        features = {
                            "time_domain": detector.extract_time_features(csi_data),
                            "frequency_domain": detector.extract_frequency_features(
                                csi_data
                            ),
                            "statistical": detector.extract_statistical_features(
                                csi_data
                            ),
                            "pca": detector.extract_pca_features(csi_data),
                        }

                        if output:
                            feature_file = os.path.join(
                                output, f"{feature_base_name}_features.npz"
                            )
                            np.savez(feature_file, **features)
                            print(f"Features saved to {feature_file}")

                    # Motion detection
                    if detect_motion:
                        window_size = 100
                        motion_events = []

                        for i in range(0, len(csi_data), window_size):
                            window = csi_data[i : i + window_size]
                            if len(window) < window_size:
                                break

                            motion_detected, confidence = detector.detect_motion(window)
                            if motion_detected:
                                motion_events.append(
                                    {
                                        "window_start": i,
                                        "confidence": confidence,
                                        "timestamp": i / 1000.0,
                                    }
                                )

                        if motion_events:
                            print(
                                f"\nDetected {len(motion_events)} motion events in {filename}:"
                            )
                            for event in motion_events:
                                print(
                                    f"  Time: {event['timestamp']:.2f}s, "
                                    f"Confidence: {event['confidence']:.2f}"
                                )

                    # Generate visualizations
                    if visualize and features:
                        visualizer.save_all_plots(
                            features, output or ".", feature_base_name
                        )
                        print(f"Visualizations saved to {output or '.'}")

                if output:
                    output_file = os.path.join(output, f"processed_{filename}")
                    np.save(output_file, csi_data)
                    print(f"Processed data saved to {output_file}")

                processed_files += 1

            except Exception as e:
                skipped_files += 1
                error_msg = f"Error processing file {filename}: {str(e)}"
                logging.error(error_msg)
                logging.error(f"Error type: {type(e).__name__}")
                if hasattr(e, "__traceback__"):
                    import traceback

                    logging.error(
                        "Traceback:\n" + "".join(traceback.format_tb(e.__traceback__))
                    )
                print(error_msg)

        logging.info(
            f"Processing complete. Processed {processed_files} files, skipped {skipped_files} files."
        )
        if skipped_files > 0:
            logging.warning(
                "Some files were skipped. Check the log file for detailed error messages."
            )
        print(f"Processing complete. Processed {processed_files} files.")
        if skipped_files > 0:
            print(f"Skipped {skipped_files} files. Check the log file for details.")

    def save_sanitized_data(
        self,
        data_dir: str,
        filename: str,
        sanitized_csi: np.ndarray,
        calib_file_path: str,
    ) -> None:
        calib_file_name = os.path.splitext(os.path.basename(calib_file_path))[0]
        output_dir = os.path.join(data_dir, f"output-pose-{calib_file_name}")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}-sanitized.npy"
        output_path = os.path.join(output_dir, output_filename)

        np.save(output_path, sanitized_csi)
        logging.info(f"Saved sanitized data to {output_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


@click.group()
def cli():
    """Process CSI data files with various operations."""
    pass


@cli.command()
@click.option("--data-dir", "-d", help="Directory containing CSI data files")
@click.option(
    "--sanitize/--no-sanitize", default=None, help="Enable sanitization of CSI data"
)
@click.option(
    "--calib-file",
    "-c",
    help="Path to calibration file (.dat or .npy format) for sanitization",
)
@click.option("--output", "-o", help="Output directory for processed data")
@click.option(
    "--save-sanitized/--no-save-sanitized",
    default=None,
    help="Save sanitized data as NumPy files",
)
@click.option(
    "--enable-logging/--no-logging",
    default=True,
    help="Enable logging to file in logs directory",
)
@click.option(
    "--nonlinear/--no-nonlinear", default=None, help="Apply nonlinear sanitization"
)
@click.option("--sto/--no-sto", default=None, help="Apply STO correction")
@click.option("--rco/--no-rco", default=None, help="Apply RCO correction")
@click.option("--cfo/--no-cfo", default=None, help="Apply CFO correction")
@click.option(
    "--extract-features/--no-features",
    default=False,
    help="Extract CSI features during processing",
)
@click.option(
    "--detect-motion/--no-motion", default=False, help="Enable motion detection"
)
@click.option(
    "--visualize/--no-visualize", default=False, help="Generate feature visualizations"
)
@click.option(
    "--method",
    type=click.Choice(["selective", "batch"]),
    default="selective",
    help="Sanitization method: selective or batch",
)
@click.option(
    "--template-linear-start",
    type=int,
    default=20,
    help="Start index for template linear interval",
)
@click.option(
    "--template-linear-end",
    type=int,
    default=39,
    help="End index for template linear interval (exclusive)",
)
def main(
    data_dir: Optional[str],
    sanitize: Optional[bool],
    calib_file: Optional[str],
    output: Optional[str],
    save_sanitized: Optional[bool],
    enable_logging: bool,
    nonlinear: Optional[bool],
    sto: Optional[bool],
    rco: Optional[bool],
    cfo: Optional[bool],
    extract_features: bool,
    detect_motion: bool,
    visualize: bool,
    method: str,
    template_linear_start: int,
    template_linear_end: int,
) -> None:
    # Set up logging if enabled
    log_file = setup_logging(enable_logging)
    if log_file:
        logging.info("Starting CSI processing")

    # Prompt for data directory if not provided
    if data_dir is None:
        data_dir = click.prompt(
            "Enter the data directory path",
            type=click.Path(exists=True, file_okay=False, dir_okay=True),
        )

    # Prompt for sanitization option if not provided
    if sanitize is None:
        sanitize = click.confirm("Enable sanitization of CSI data?", default=False)

    # If sanitization is enabled, prompt for calibration file and sanitization options
    if sanitize:
        if calib_file is None:
            calib_file = click.prompt(
                "Enter the calibration file path",
                type=click.Path(exists=True, file_okay=True, dir_okay=False),
            )

        if nonlinear is None:
            nonlinear = click.confirm("Apply nonlinear sanitization?", default=True)

        if sto is None:
            sto = click.confirm("Apply STO correction?", default=True)

        if rco is None:
            rco = click.confirm("Apply RCO correction?", default=True)

        if cfo is None:
            cfo = click.confirm("Apply CFO correction?", default=False)

        if save_sanitized is None:
            save_sanitized = click.confirm(
                "Save sanitized data as NumPy files?", default=False
            )

    # Prompt for output directory if not provided
    if output is None:
        output = click.prompt(
            "Enter the output directory for processed data (optional)",
            default="",
            show_default=False,
        )
        if output == "":
            output = None

    processor = CSIProcessor()
    processor.process_csi_data(
        data_dir,
        sanitize,
        calib_file,
        output,
        save_sanitized,
        nonlinear,
        sto,
        rco,
        cfo,
        extract_features,
        detect_motion,
        visualize,
        method,
        template_linear_start,
        template_linear_end,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
