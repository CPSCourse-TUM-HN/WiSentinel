# RSSI-Based Indoor Localization

This project uses Wi-Fi signal strength (RSSI) to estimate indoor positions via two methods: **Trilateration** and **Fingerprinting**.

## Quickstart Guide

### Step 1: Setup

1. **Define Survey Grid**: Modify and run the `grid_points.csv` generator script to create `grid_points.csv` for your area.

   ```python
   # grid_points_generator.py
   import csv
   import numpy as np
   x_max, y_max, spacing = 6.0, 4.0, 1.0 # Edit these (in meters)
   x_coords = np.arange(0, x_max + 0.001, spacing)
   y_coords = np.arange(0, y_max + 0.001, spacing)
   with open('../Localized_Engine/grid_points.csv', 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['x', 'y', 'point_name'])
       for j, y in enumerate(y_coords):
           for i, x in enumerate(x_coords):
               writer.writerow([f"{x:.1f}", f"{y:.1f}", f"pt_{i}{j}"])
   ```

2.  **Create `anchors.json`**: Find your Wi-Fi APs' MAC addresses (`list_aps.py` on macOS or `sudo iw dev wlan0 scan` on Linux) and list them in an `anchors.json` file.

    ```json
    [
      { "mac": "a8:93:4a:df:a6:c1", "alias": "AP1" },
      { "mac": "38:d5:7a:a4:94:f7", "alias": "AP2" }
    ]
    ```

### Step 2: Collect & Process Data

1.  **Log RSSI Data (Linux)**: At each grid point, run the logger to collect signal strengths.

    ```bash
    sudo python3 rssi_logger.py \
      --iface <wlan0> --anchors anchors.json --grid grid_points.csv --output rssi_log.csv
    ```

2.  **Compute Means**: Process the raw logs into a clean dataset.

    ```bash
    python3 compute_rssi_means.py --input rssi_log.csv --output rssi_means.csv
    ```

### Step 3: Calibrate & Localize

1.  **Estimate Anchor Positions**: Automatically generate anchor coordinates for trilateration.

    ```bash
    python3 anchor_calibration.py --means rssi_means.csv --output anchors_est.json
    ```

2.  **Run Localization**: Evaluate the system using either method.

      - **Trilateration**:

        ```bash
        python3 rssi_localization.py --anchors anchors_est.json --means rssi_means.csv --mode trilat
        ```

      - **Fingerprinting**:

        ```bash
        python3 rssi_localization.py --anchors anchors_est.json --means rssi_means.csv --mode fingerprint -k 3
        ```

The script will print accuracy metrics (MAE, RMSE) to the console.

## How It Works

  - **Trilateration**: Converts RSSI to distance using a path-loss model ($RSSI = P\_0 - 10n \\log\_{10}(d)$) and finds the geometric intersection of distance circles from multiple anchors.
  - **Fingerprinting**: Creates a "radio map" of RSSI values at known locations. It then finds the best match for a new signal by comparing it to the map using a K-Nearest Neighbors (KNN) algorithm.