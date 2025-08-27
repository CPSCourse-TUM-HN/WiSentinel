#!/usr/bin/env python3
"""
rssi_logger.py

Scans a Linux Wi-Fi interface using `iw dev <iface> scan` to collect RSSI (dBm) for specified anchors at grid points.

Usage:
  sudo ./rssi_logger.py --iface wlan-ath --anchors BSSID.json --grid grid_points.csv --output rssi_log.csv --interval 1.0

Outputs:
  CSV with columns: timestamp,point_name,x,y,alias,mac,rssi
"""
import argparse
import csv
import datetime
import json
import subprocess
import time
import re


def scan_rssi(mac_list, iface):
    """
    Scan with `iw dev iface scan` and return dict {mac: rssi_dBm} for macs in mac_list.
    Parses 'BSS <mac>' and 'signal: <value> dBm' lines robustly.
    """
    bss_re = re.compile(r"^BSS\s+([0-9A-Fa-f:]{17})")
    sig_re = re.compile(r"signal:\s*(-?\d+\.?\d*)\s*dBm", re.IGNORECASE)

    proc = subprocess.run(
        ["sudo", "iw", "dev", iface, "scan"], capture_output=True, text=True
    )
    lines = proc.stdout.splitlines()

    rssi_dict = {}
    current_bss = None
    macs_lower = [m.lower() for m in mac_list]

    for line in lines:
        line = line.strip()
        m_bss = bss_re.match(line)
        if m_bss:
            current_bss = m_bss.group(1).lower()
            continue
        if current_bss:
            m_sig = sig_re.search(line)
            if m_sig:
                try:
                    val = float(m_sig.group(1))
                    rssi = int(round(val))
                except ValueError:
                    current_bss = None
                    continue
                if current_bss in macs_lower:
                    orig = mac_list[macs_lower.index(current_bss)]
                    rssi_dict[orig] = rssi
                current_bss = None
    return rssi_dict


def main():
    parser = argparse.ArgumentParser(description="RSSI Logger using iw")
    parser.add_argument(
        "--iface", required=True, help="Wi-Fi interface (e.g., wlan-ath)"
    )
    parser.add_argument("--anchors", required=True, help="Path to anchors JSON file")
    parser.add_argument("--grid", required=True, help="Path to grid points CSV")
    parser.add_argument("--output", required=True, help="Output CSV for RSSI logs")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between scans per grid point",
    )
    args = parser.parse_args()

    # Load anchors
    with open(args.anchors) as f:
        anchors = json.load(f)
    mac_list = [a["mac"] for a in anchors]
    alias_map = {a["mac"]: a.get("alias", a["mac"]) for a in anchors}

    # Load grid points
    grid_points = []
    with open(args.grid) as f:
        reader = csv.DictReader(f)
        for row in reader:
            grid_points.append((float(row["x"]), float(row["y"]), row["point_name"]))

    # Open output CSV
    with open(args.output, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["timestamp", "point_name", "x", "y", "alias", "mac", "rssi"])

        # Iterate over grid points
        for x, y, pname in grid_points:
            print(f"Collecting at {pname} ({x},{y})…")
            start = time.time()
            while time.time() - start < 10:
                ts = datetime.datetime.utcnow().isoformat()
                readings = scan_rssi(mac_list, args.iface)
                print(f"[DEBUG] {pname} -> {readings}")
                for mac, rssi in readings.items():
                    writer.writerow([ts, pname, x, y, alias_map[mac], mac, rssi])
                out_f.flush()
                time.sleep(args.interval)
            print(f"Finished {pname}")


if __name__ == "__main__":
    main()
