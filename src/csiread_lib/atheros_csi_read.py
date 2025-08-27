"""csiread.Atheros: implemented in pure Python with debug outputs"""

import numpy as np
import os
from timeit import default_timer
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


class Atheros:
    def __init__(self, file, nrxnum=3, ntxnum=3, tones=56, pl_len=0, if_report=True):
        """Parameter initialization."""
        self.file = file
        self.nrxnum = nrxnum
        self.ntxnum = ntxnum
        self.tones = tones
        self.pl_len = pl_len
        self.if_report = if_report

        if not os.path.isfile(file):
            raise Exception("error: file does not exist, Stop!\n")

        print(f"\n[DEBUG] Initialized Atheros CSI reader with:")
        print(f"  File: {file}")
        print(f"  NRx: {nrxnum}, NTx: {ntxnum}, Tones: {tones}")
        print(f"  Payload length: {pl_len}, Reporting: {if_report}\n")

    def read(self, endian="big"):
        print(
            f"[DEBUG] Opening file {self.file} in {'little' if endian == 'little' else 'big'} endian mode"
        )
        f = open(self.file, "rb")
        if f is None:
            f.close()
            return -1

        lens = os.path.getsize(self.file)
        print(f"[DEBUG] File size: {lens} bytes")

        btype = np.intp
        num_packets = lens // 420
        print(f"[DEBUG] Allocating arrays for ~{num_packets} packets")

        # Initialize all arrays
        self.timestamp = np.zeros([num_packets])
        self.csi_len = np.zeros([num_packets], dtype=btype)
        self.tx_channel = np.zeros([num_packets], dtype=btype)
        self.err_info = np.zeros([num_packets], dtype=btype)
        self.noise_floor = np.zeros([num_packets], dtype=btype)
        self.Rate = np.zeros([num_packets], dtype=btype)
        self.bandWidth = np.zeros([num_packets], dtype=btype)
        self.num_tones = np.zeros([num_packets], dtype=btype)
        self.nr = np.zeros([num_packets], dtype=btype)
        self.nc = np.zeros([num_packets], dtype=btype)
        self.rssi = np.zeros([num_packets], dtype=btype)
        self.rssi_1 = np.zeros([num_packets], dtype=btype)
        self.rssi_2 = np.zeros([num_packets], dtype=btype)
        self.rssi_3 = np.zeros([num_packets], dtype=btype)
        self.payload_len = np.zeros([num_packets], dtype=btype)
        self.csi = np.zeros(
            [num_packets, self.tones, self.nrxnum, self.ntxnum], dtype=np.complex128
        )
        self.payload = np.zeros([num_packets, self.pl_len], dtype=btype)

        cur = 0
        count = 0
        print("\n[DEBUG] Starting packet processing...")

        while cur < (lens - 4):
            if count % 100 == 0:
                print(f"[DEBUG] Processing packet {count} at offset {cur}/{lens}")

            field_len = int.from_bytes(f.read(2), byteorder=endian)
            cur += 2
            if (cur + field_len) > lens:
                print(
                    f"[WARNING] Packet {count} would exceed file size (field_len={field_len})"
                )
                break

            # Read all packet fields
            self.timestamp[count] = int.from_bytes(f.read(8), byteorder=endian)
            cur += 8
            self.csi_len[count] = int.from_bytes(f.read(2), byteorder=endian)
            cur += 2
            self.tx_channel[count] = int.from_bytes(f.read(2), byteorder=endian)
            cur += 2
            self.err_info[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.noise_floor[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.Rate[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.bandWidth[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.num_tones[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.nr[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.nc[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.rssi[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.rssi_1[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.rssi_2[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.rssi_3[count] = int.from_bytes(f.read(1), byteorder=endian)
            cur += 1
            self.payload_len[count] = int.from_bytes(f.read(2), byteorder=endian)
            cur += 2

            # Debug output for packet headers
            if count < 3:  # Print first 3 packets' headers for verification
                print(f"\n[DEBUG] Packet {count} header:")
                print(f"  Timestamp: {self.timestamp[count]}")
                print(f"  CSI len: {self.csi_len[count]}")
                print(f"  TX Channel: {self.tx_channel[count]}")
                print(f"  NR: {self.nr[count]}, NC: {self.nc[count]}")
                print(f"  Num tones: {self.num_tones[count]}")
                print(f"  Payload len: {self.payload_len[count]}")

            # Process CSI data
            c_len = self.csi_len[count]
            if c_len > 0:
                csi_buf = f.read(c_len)
                try:
                    self.csi[count] = self.__read_csi(
                        csi_buf, self.nr[count], self.nc[count], self.num_tones[count]
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process CSI for packet {count}: {str(e)}")
                    print(
                        f"  NR: {self.nr[count]}, NC: {self.nc[count]}, Tones: {self.num_tones[count]}"
                    )
                    raise
                cur += c_len
            else:
                self.csi[count] = None

            # Process payload
            pl_len = self.payload_len[count]
            pl_stop = min(pl_len, self.pl_len, 0)
            if pl_len > 0:
                self.payload[count, :pl_stop] = bytearray(f.read(pl_len))[:pl_stop]
                cur += pl_len
            else:
                self.payload[count, :pl_stop] = 0

            if cur + 420 > lens:
                count -= 1
                print(f"[DEBUG] Reached end of file at packet {count}")
                break
            count += 1

        # Trim arrays to actual packet count
        actual_count = count
        print(f"\n[DEBUG] Processed {actual_count} complete packets")

        self.timestamp = self.timestamp[:actual_count]
        self.csi_len = self.csi_len[:actual_count]
        self.tx_channel = self.tx_channel[:actual_count]
        self.err_info = self.err_info[:actual_count]
        self.noise_floor = self.noise_floor[:actual_count]
        self.Rate = self.Rate[:actual_count]
        self.bandWidth = self.bandWidth[:actual_count]
        self.num_tones = self.num_tones[:actual_count]
        self.nr = self.nr[:actual_count]
        self.nc = self.nc[:actual_count]
        self.rssi = self.rssi[:actual_count]
        self.rssi_1 = self.rssi_1[:actual_count]
        self.rssi_2 = self.rssi_2[:actual_count]
        self.rssi_3 = self.rssi_3[:actual_count]
        self.payload_len = self.payload_len[:actual_count]
        self.csi = self.csi[:actual_count]
        self.payload = self.payload[:actual_count]

        f.close()
        print("[DEBUG] File processing completed successfully")

    def remove_nan_packets(self):
        """Remove packets with NaN values in CSI data."""
        if self.csi is None:
            return

        valid_mask = ~np.isnan(self.csi).any(axis=(1, 2, 3))
        original_count = len(self.csi)
        self.csi = self.csi[valid_mask]
        self.timestamp = self.timestamp[valid_mask]
        self.csi_len = self.csi_len[valid_mask]
        self.tx_channel = self.tx_channel[valid_mask]
        self.err_info = self.err_info[valid_mask]
        self.noise_floor = self.noise_floor[valid_mask]
        self.Rate = self.Rate[valid_mask]
        self.bandWidth = self.bandWidth[valid_mask]
        self.num_tones = self.num_tones[valid_mask]
        self.nr = self.nr[valid_mask]
        self.nc = self.nc[valid_mask]
        self.rssi = self.rssi[valid_mask]
        self.rssi_1 = self.rssi_1[valid_mask]
        self.rssi_2 = self.rssi_2[valid_mask]
        self.rssi_3 = self.rssi_3[valid_mask]
        self.payload_len = self.payload_len[valid_mask]
        print(f"[DEBUG] Removed {original_count - len(self.csi)} packets with NaN values")

    def __read_csi(self, csi_buf, nr, nc, num_tones):
        # Add safety checks
        nr = min(nr, self.nrxnum)
        nc = min(nc, self.ntxnum)
        num_tones = min(num_tones, self.tones)

        #print(f"[DEBUG] Reading CSI: NR={nr}, NC={nc}, Tones={num_tones}, Buffer_len={len(csi_buf)}")

        csi = np.zeros([self.tones, self.nrxnum, self.ntxnum], dtype=np.complex128)
        bits_left = 16
        bitmask = (1 << 10) - 1

        idx = 0
        h_data = csi_buf[idx]
        idx += 1
        h_data += csi_buf[idx] << 8
        idx += 1
        curren_data = h_data & ((1 << 16) - 1)

        for k in range(num_tones):
            for nc_idx in range(nc):
                for nr_idx in range(nr):
                    # imag
                    if (bits_left - 10) < 0:
                        h_data = csi_buf[idx]
                        idx += 1
                        h_data += csi_buf[idx] << 8
                        idx += 1
                        curren_data += h_data << bits_left
                        bits_left += 16
                    imag = curren_data & bitmask
                    imag = self.__signbit_convert(imag, 10)

                    bits_left -= 10
                    curren_data = curren_data >> 10
                    # real
                    if (bits_left - 10) < 0:
                        h_data = csi_buf[idx]
                        idx += 1
                        h_data += csi_buf[idx] << 8
                        idx += 1
                        curren_data += h_data << bits_left
                        bits_left += 16
                    real = curren_data & bitmask
                    real = self.__signbit_convert(real, 10)

                    bits_left -= 10
                    curren_data = curren_data >> 10
                    # csi
                    csi[k, nr_idx, nc_idx] = real + imag * 1j

        return csi

    def __signbit_convert(self, data, maxbit):
        if data & (1 << (maxbit - 1)):
            data -= 1 << maxbit
        return data


def visualize_csi(csidata: "Atheros", title: str = "CSI Data") -> None:
    """Visualize CSI data with statistics and plots"""
    print("\n=== CSI Validation ===")

    # Print CSI statistics
    print(f"\nCSI Stats:")
    print(f"  Shape: {csidata.csi.shape} (Packets, Tones, NRx, NTx)")
    print(f"  Mean magnitude: {np.abs(csidata.csi).mean():.2f}")
    print(f"  Std dev: {np.abs(csidata.csi).std():.2f}")

    # Plot magnitude over time for first TX-RX pair
    plt.figure(figsize=(14, 8))
    plt.plot(np.abs(csidata.csi[:, :, 0, 0]))
    plt.title(f"Subcarrier Magnitudes Over Time (TX0-RX0)\nFile: {csidata.file}")
    plt.xlabel("Packet Index")
    plt.ylabel("Magnitude")

    # Update the title of the plot
    plt.suptitle(title)

    plt.show()


def main(csi_file: str = None) -> None:
    """Process and visualize CSI data from file"""
    print("=== CSI Data Reader ===")
    last = default_timer()

    if csi_file is None:
        csi_file = sys.argv[1] if len(sys.argv) > 1 else "csi_data.dat"
        print(
            "Write filename after the script name, e.g. 'python atheros_csi_read.py csi_data.dat'"
        )

    try:
        print(f"\n[MAIN] Creating Atheros reader for {csi_file}")
        csidata = Atheros(csi_file, nrxnum=3, ntxnum=3, pl_len=10, if_report=True)

        print("[MAIN] Reading CSI data...")
        csidata.read()

        print("\n[MAIN] Summary:")
        print(f"  Processed {len(csidata.timestamp)} packets")
        print(f"  CSI shape: {csidata.csi.shape}")
        print(f"  Execution time: {default_timer() - last:.2f} seconds")

        # Visualize the data
        visualize_csi(csidata)

        return csidata

    except Exception as e:
        print(f"\n[ERROR] Failed to process CSI data: {str(e)}")
        raise


if __name__ == "__main__":
    main()
