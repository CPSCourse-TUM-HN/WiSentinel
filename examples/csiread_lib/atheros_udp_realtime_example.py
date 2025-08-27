import time
import logging
import numpy as np
from csiread_lib import AtherosUDPRealtime

# Step 1: Configure root logger to see output from your class
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for maximum verbosity
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Step 2: Initialize the CSI receiver
receiver = AtherosUDPRealtime(
    port=8000,
    addr="0.0.0.0",
    nrxnum=3,
    ntxnum=3,
    tones=56,
    max_packets=200,
    log_level=logging.DEBUG,  # Controls verbosity of this instance's logger
)

# Step 3: Start the background UDP receiver
receiver.start()

# Step 4: Simulate periodic processing loop
try:
    print("📡 Receiving CSI... (press Ctrl+C to stop)\n")

    while True:
        time.sleep(
            1
        )  # Simulate processing loop at 1 Hz (e.g. visualization update rate)

        # Drain up to N packets (None = all available)
        packets = receiver.drain_buffer()

        if not packets:
            print("[WAIT] No packets yet.")
            continue

        print(f"[INFO] Received {len(packets)} new packets")

        # Just inspect the shape and type of first packet
        csi = packets[0]
        print(f"↳ First packet shape: {csi.shape}, dtype: {csi.dtype}")

        # Example: Print average magnitude for first packet
        magnitude = np.abs(csi)
        print(f"↳ Avg CSI magnitude (first packet): {magnitude.mean():.2f}\n")

except KeyboardInterrupt:
    print("\n👋 Shutting down gracefully...")

finally:
    # Step 5: Stop background thread and close socket
    receiver.stop()
    print("✅ Done.")
