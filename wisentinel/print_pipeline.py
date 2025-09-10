# This script
# Reads .dat file via AtherosReader class
# Aggregates first 300 packets into 60 packet windows
# Prints first 5 averaged windows in form of 3D tensors = 56 subcarriers x 3 rx antennas x 3 tx antennas
# Runs sanitization
# Prints same first 5 averaged windows packets after sanitization
# Extracts features from all 60 packet windows
# Prints first 5 feature vectors (5 should be specifiable in the variable)
# .dat filename specifiable as command line argument

import numpy as np
import pandas as pd
from csiread_lib.atheros_csi_read import Atheros
from feature_extraction import FeatureExtractor
from sanitization import SanitizeDenoising, load_csi_data
import sys

def main():
    # Filename of the .dat file from command line argument
    dat_filename = sys.argv[1]
    reader = Atheros(dat_filename)
    reader.read()
    reader.remove_nan_packets()
    csi_data = reader.csi
    print(f"Read {len(csi_data)} packets from {dat_filename}")


    # Sanitization
    
    calib_path = "../dataset/ilive/no_person/ilive-empty-room-1min.dat"  
    calib_csi, _ = load_csi_data(calib_path)

    print (f"FIRST PACKET:")
    print(f"Packet {0} first 2 subcarriers\n: {csi_data[0][:2]}")

    sanitizer = SanitizeDenoising(calib_path)
    linear_interval = np.arange(20, 39)
    template = sanitizer.set_template(calib_csi, linear_interval)

    # Initialize sanitizer
    print(f"SANITIZING {len(csi_data)} PACKETS...")

    sanitized_packets = []
    sanitized_csi = sanitizer.sanitize_csi_selective(
                csi_data, template, nonlinear=True, sto=True, rco=True, cfo=False
    )

    print(f"FIRST SANITIZED PACKET:")
    print(f"Sanitized packets shape: {sanitized_csi.shape}")

    print(f"Sanitized Packet {0}, first 2 subcarriers\n: {sanitized_csi[0][:2]}")

if __name__ == '__main__':
    main()