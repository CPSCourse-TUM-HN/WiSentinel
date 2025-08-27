# Hardware Setup

## Firmware Building

### Building Custom OpenWRT Image

#### 1. Setup Docker Environment
```bash
# Build the Docker image
cd hardware_setup/build_custom
docker build -t openwrt-csi-builder .

# Run the Docker container
docker run -it --name openwrt-build openwrt-csi-builder
```

#### 2. Build OpenWRT
Inside the Docker container:
```bash
# Update feeds
./scripts/feeds update -a
./scripts/feeds install -a

# Configure build
make menuconfig
```

Important menuconfig settings:
- Target System: `Atheros AR7xxx/AR9xxx`
- Subtarget: `Generic devices with NAND flash`
- Target Profile: `TP-Link Archer C7 v5`
- Enable CSI collection patches under `Kernel modules → Wireless Drivers`

```bash
# Start the build process
make -j$(nproc)
```

#### 3. Extract Built Images
After the build completes:
```bash
# From your host machine
docker cp openwrt-build:/openwrt/bin/targets/ar71xx/generic/openwrt-ar71xx-generic-archer-c7-v5-squashfs-factory.bin .
docker cp openwrt-build:/openwrt/bin/targets/ar71xx/generic/openwrt-ar71xx-generic-archer-c7-v5-squashfs-sysupgrade.bin .
```

### Available Images
The build process generates two images:

1. **Factory Image** (for fresh installation):
   ```
   openwrt-ar71xx-generic-archer-c7-v5-squashfs-factory.bin
   ```
   Use this image when flashing OpenWRT for the first time from stock firmware.

2. **Upgrade Image** (for existing OpenWRT):
   ```
   openwrt-ar71xx-generic-archer-c7-v5-squashfs-sysupgrade.bin
   ```
   Use this image when upgrading from an existing OpenWRT installation.

## Router Setup

## Devices Used
The system utilizes two TP-Link Archer C7 v2 routers, organized into workstation set. Set consists of:
- 1 router configured as CSI receiver (`recvCSI`)
- 1 router configured as CSI sender and access point (`sendData`)

All routers run Atheros CSI-patched OpenWRT firmware (based on version 22.03.05) and share the following credentials:
- **Username:** root
- **Password:** ccna
- **SSH Access:** `ssh -oHostKeyAlgorithms=+ssh-rsa root@192.168.1.11`

## Network Topology
| Device          | IP Address    | MAC Address       | Role                     |
|-----------------|--------------|-------------------|--------------------------|
| OpenWRT1        | 192.168.1.11 | 98:de:d0:a4:66:68 | CSI Receiver (recvCSI)   |
| OpenWRT2        | 192.168.1.12 | 7c:8b:ca:d4:49:f2 | CSI Sender/AP (sendData) |

**Infrastructure Notes:**
- Both routers connect to the same switch for Ethernet fallback access
- Workstation/laptop connects via SSH for control and analysis

## Configuration Guide

### 1. Configure OpenWRT2 as CSI Access Point
SSH into OpenWRT2 (192.168.1.12) and execute:
```bash
uci set wireless.@wifi-device[0].channel='6'
uci set wireless.@wifi-iface[0].mode='ap'
uci set wireless.@wifi-iface[0].ssid='csitest'
uci set wireless.@wifi-iface[0].encryption='none'
uci set wireless.@wifi-iface[0].network='lan'
uci set wireless.radio0.disabled='0'
uci commit wireless
wifi reload
```

Start continuous CSI transmission:
```bash
sendData
```

### 2. Configure OpenWRT1 as CSI Receiver

#### A. Connect to Access Point
SSH into OpenWRT1 (192.168.1.11):
```bash
uci set wireless.@wifi-iface[0].mode='sta'
uci set wireless.@wifi-iface[0].ssid='csitest'
uci set wireless.@wifi-iface[0].encryption='none'
uci set wireless.radio0.disabled='0'
uci commit wireless
wifi reload
```

Verify connection:
```bash
iw dev wlan0 link
```

#### B. Create Monitor Interface
```bash
iw dev wlan0 interface add mon0 type monitor
ifconfig mon0 up
```

Confirm monitor interface status:
```bash
iw dev mon0 info
```

### 3. Initiate CSI Capture
Begin data collection on OpenWRT1:
```bash
recvCSI /tmp/csi_output.dat
```

Successful operation will display real-time CSI capture messages.

---

**Maintenance Notes:**
- All configuration changes require `uci commit` and `wifi reload` to take effect
- Monitor interface (`mon0`) must be recreated after router reboot
- For consistent results, maintain clear line-of-sight between routers during operation

This documentation reflects the tested configuration as of OpenWRT 22.03.05 with Atheros CSI patches.