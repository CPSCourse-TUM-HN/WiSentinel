#!/usr/bin/env python3
"""
list_aps.py

Scan for nearby Wi-Fi networks using CoreWLAN and list SSID, BSSID, and RSSI.
"""
from CoreWLAN import CWInterface


def list_networks():
    iface = CWInterface.interface()
    networks, err = iface.scanForNetworksWithSSID_error_(None, None)
    if err:
        print("❌ Scan error:", err)
        return
    print(f"{'SSID':20} {'BSSID':20} {'RSSI'}")
    for net in networks:
        ssid = net.ssid() or "<hidden>"
        bssid = net.bssid() or ""
        try:
            rssi = net.rssiValue()
        except Exception:
            rssi = None
        print(f"{ssid:20} {bssid:20} {rssi}")


if __name__ == "__main__":
    list_networks()
