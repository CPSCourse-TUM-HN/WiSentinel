import sys
import os
from pathlib import Path

# Ensure src/cli is in sys.path for direct import
cli_dir = Path(__file__).parent
if str(cli_dir) not in sys.path:
    sys.path.insert(0, str(cli_dir))

import cli_static
import cli_live

# TODO: make a live listening version
# TODO: check the pmsg() function for real time packet reading.

# file is put in the data_files folder for testing purposes
# source: https://github.com/Gi-z/CSI-Data
# source: https://github.com/citysu/csiread
# can be uncommented to test a CLI client.


def main():
    print("Select CSI processing mode:")
    print("1. Real-time CLI (live streaming)")
    print("2. Static processing CLI (batch or selective file processing)")
    while True:
        choice = input("> ").strip()
        if choice == "1":
            cli_live.main()
            break
        elif choice == "2":
            cli_static.main()
            break
        else:
            print("Invalid choice. Please select 1 or 2.")
