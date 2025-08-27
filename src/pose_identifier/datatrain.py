import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import struct
import os
from collections import defaultdict

MODE = "PREPARE_DATA"
WINDOW_SIZE = 100  # Define window size globally


def read_csi_dat(file_path: str) -> np.ndarray:
    # ... (no changes to this function)
    csi_data = []
    num_tx_antennas = 1
    num_rx_antennas = 3
    num_subcarriers = 30
    try:
        with open(file_path, "rb") as f:
            while True:
                field_len_bytes = f.read(2)
                if not field_len_bytes:
                    break
                field_len = struct.unpack(">H", field_len_bytes)[0]
                csi_bytes = f.read(field_len)
                if len(csi_bytes) != field_len:
                    break
                num_complex_vals = num_tx_antennas * num_rx_antennas * num_subcarriers
                dt = np.dtype(np.int8)
                csi_raw = np.frombuffer(
                    csi_bytes, dtype=dt, count=num_complex_vals * 2, offset=10
                )
                if csi_raw.size < num_complex_vals * 2:
                    continue
                csi_cmplx = csi_raw[0::2].astype(np.float32) + 1j * csi_raw[
                    1::2
                ].astype(np.float32)
                csi_matrix = csi_cmplx.reshape(
                    num_tx_antennas, num_rx_antennas, num_subcarriers
                )
                csi_data.append(csi_matrix)
    except Exception as e:
        print(f"  [ERROR] reading {file_path}: {e}")
        return np.array([])
    return np.array(csi_data)


def calculate_dfs_profile(
    csi_series: np.ndarray, packet_rate: int, two_sided_spectrum: bool = True
):
    # ... (no changes to this function)
    if csi_series.size == 0:
        return None, None, None
    csi_series = np.squeeze(csi_series, axis=1)
    rx1 = csi_series[:, 0, :]
    rx2 = csi_series[:, 1, :]
    subcarrier_index = 15
    csi_rx1_sc = rx1[:, subcarrier_index]
    csi_rx2_sc = csi_rx1_sc * np.conj(rx2[:, subcarrier_index])
    phase_series = np.unwrap(np.angle(csi_rx2_sc))
    f, t, Zxx = stft(
        phase_series, fs=packet_rate, nperseg=WINDOW_SIZE, return_onesided=False
    )
    if two_sided_spectrum:
        Zxx = np.fft.fftshift(Zxx, axes=0)
        f = np.fft.fftshift(f)
    return f, t, np.abs(Zxx)


def generate_gesture_plot(
    gesture_prefix: str, file_paths: list, packet_rate: int, output_dir: str
):
    # ... (no changes to this function)
    print(f"\n--- Plotting Gesture: {gesture_prefix} ---")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    fig.suptitle(f"DFS Spectrograms for Gesture: {gesture_prefix}", fontsize=16)
    axes = axes.flatten()
    for i, file_path in enumerate(sorted(file_paths)):
        csi_data = read_csi_dat(file_path)
        if csi_data.shape[0] < WINDOW_SIZE:
            print(f"  [SKIPPING] File {os.path.basename(file_path)} is too short.")
            continue
        if csi_data.size > 0:
            f, t, Zxx = calculate_dfs_profile(
                csi_data, packet_rate, two_sided_spectrum=True
            )
            if f is not None:
                ax = axes[i]
                ax.pcolormesh(t, f, Zxx, shading="gouraud")
                ax.set_title(f"Receiver {i + 1}")
                ax.set_ylim(-100, 100)
    for ax in axes:
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{gesture_prefix}.png")
    plt.savefig(save_path)
    print(f"  Plot saved to: {save_path}")
    plt.close(fig)


def prepare_gesture_data(
    gesture_prefix: str, file_paths: list, packet_rate: int, output_dir: str
):
    """Processes receiver files for a gesture, stacks them, and saves as a .npy file for training."""
    print(f"\n--- Preparing Data for Gesture: {gesture_prefix} ---")
    all_dfs_profiles = []

    # --- THE FIX: Validate all files in the group before processing ---
    for file_path in sorted(file_paths):
        csi_data = read_csi_dat(file_path)
        if csi_data.shape[0] < WINDOW_SIZE:
            print(
                f"  [SKIPPING GROUP] Gesture {gesture_prefix} contains a short file: {os.path.basename(file_path)}. Discarding this entire gesture sample."
            )
            return  # Exit this function, skipping the gesture

        _, _, Zxx = calculate_dfs_profile(
            csi_data, packet_rate, two_sided_spectrum=False
        )
        if Zxx is not None:
            all_dfs_profiles.append(Zxx)

    if all_dfs_profiles and len(all_dfs_profiles) == len(file_paths):
        max_time_steps = max(zxx.shape[1] for zxx in all_dfs_profiles)
        padded_dfs_profiles = []
        for zxx in all_dfs_profiles:
            pad_width = max_time_steps - zxx.shape[1]
            padded_zxx = np.pad(
                zxx, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
            )
            padded_dfs_profiles.append(padded_zxx)

        training_sample = np.stack(padded_dfs_profiles, axis=-1)
        sample_save_path = os.path.join(output_dir, f"{gesture_prefix}_data.npy")
        np.save(sample_save_path, training_sample)
        print(f"  Saved training data to: {os.path.basename(sample_save_path)}")
    else:
        print(
            f"  [SKIPPING GROUP] Gesture {gesture_prefix} could not be fully processed."
        )


# --- Main Execution ---
if __name__ == "__main__":
    # ... (no changes to this block)
    root_data_dir = "./CSI_data/"
    PACKET_RATE_HZ = 1000
    if MODE == "PREPARE_DATA":
        output_dir = "./Training_Data/"
        print(f"--- MODE: PREPARE_DATA ---")
    elif MODE == "GENERATE_PLOTS":
        output_dir = "./DFS_Spectrograms/"
        print(f"--- MODE: GENERATE_PLOTS ---")
    else:
        raise ValueError(
            "Invalid MODE selected. Choose 'PREPARE_DATA' or 'GENERATE_PLOTS'."
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    print(f"--- Grouping files by gesture ---")
    gesture_groups = defaultdict(list)
    for subdir, dirs, files in os.walk(root_data_dir):
        for filename in files:
            if filename.endswith(".dat"):
                prefix = "-".join(filename.split("-")[:-1])
                gesture_groups[prefix].append(os.path.join(subdir, filename))
    if not gesture_groups:
        print("[CRITICAL] No .dat files were found.")
    else:
        print(f"Found {len(gesture_groups)} unique gestures. Starting processing...")
        for prefix, files in gesture_groups.items():
            if MODE == "PREPARE_DATA":
                prepare_gesture_data(prefix, files, PACKET_RATE_HZ, output_dir)
            elif MODE == "GENERATE_PLOTS":
                generate_gesture_plot(prefix, files, PACKET_RATE_HZ, output_dir)
        print("\n--- All files processed. ---")
