# .dat file specifiable by command line argument
# Reads CSI data from a file, prints only final statistics

import sys
import numpy as np
from csiread_lib.atheros_csi_read import Atheros

def main():
    # Get .dat file from command line argument
    if len(sys.argv) < 2:
        print("Usage: python atheros_csi_read_from_file.py <file.dat>")
        sys.exit(1)
    dat_file = sys.argv[1]

    # Read CSI data
    csi_reader = Atheros(dat_file)
    csi_reader.read()
    csi_reader.remove_nan_packets()

    csi = csi_reader.csi
    num_packets = csi.shape[0]

    correct_shape = (56, 3, 3)
    packets_correct_shape = 0
    packets_incorrect_shape = 0
    packets_with_nan = 0
    max_consecutive_nan = 0
    current_consecutive_nan = 0

    nan_packets_info = []

    for i in range(num_packets):
        pkt = csi[i]
        # Check for correct shape
        if pkt is not None and pkt.shape == correct_shape:
            packets_correct_shape += 1
        else:
            packets_incorrect_shape += 1

        # Check for NaN values
        if pkt is not None and np.isnan(pkt).any():
            packets_with_nan += 1
            current_consecutive_nan += 1
            if current_consecutive_nan > max_consecutive_nan:
                max_consecutive_nan = current_consecutive_nan

            # Save info about NaN ratios
            nan_count = np.isnan(pkt).sum()
            total_count = pkt.size
            nan_ratio = nan_count / total_count
            nan_packets_info.append((i, nan_ratio))
        else:
            current_consecutive_nan = 0

    print("=== CSI Data Statistics ===")
    print(f"Total packets read: {num_packets}")
    print(f"Packets with correct shape {correct_shape}: {packets_correct_shape}")
    print(f"Packets with incorrect shape: {packets_incorrect_shape}")
    print(f"Packets with NaN values: {packets_with_nan}")
    print(f"Biggest number of consecutive packets with NaN values: {max_consecutive_nan}")

    if nan_packets_info:
        print("\n=== NaN Ratios for packets with NaN values ===")
        for idx, nan_ratio in nan_packets_info:
            print(f"Packet {idx}: NaN ratio = {nan_ratio:.4f} ({nan_ratio*100:.2f}%)")

if __name__ == "__main__":
    main()