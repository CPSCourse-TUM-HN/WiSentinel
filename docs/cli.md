# WiSentinel CLI Guide

This guide covers how to use WiSentinel's command-line interface (CLI) tools for both live and static CSI data processing.

## Process a directory of CSI files
python src/cli/cli_static.py process \
    --data-dir /path/to/csi/files \
    --output /path/to/output

# Process with feature detection and visualization
python src/cli/cli_static.py process \
    --data-dir /path/to/csi/files \
    --output /path/to/output \
    --extract-features \     # Enable feature extraction
    --detect-motion \       # Enable motion detection
    --visualize            # Generate visualizations

# Extract features
python src/cli/cli_static.py extract-features \
    /path/to/csi_file.dat \
    /path/to/output

# Detect motion
python src/cli/cli_static.py detect-motion \
    /path/to/csi_file.datend using `pyenv` to manage Python versions and virtual environments for a consistent development experience.

### 1. Install pyenv (Recommended)

```bash
# Linux/Mac
curl https://pyenv.run | bash

# Add to your shell configuration (~/.bashrc, ~/.zshrc, etc.):
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Windows (using pyenv-win)
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

### 2. Install Python and Create Environment

```bash
# Install Python 3.12 (or your preferred version)
pyenv install 3.12

# Create a new pyenv virtualenv
pyenv virtualenv 3.12 wisentinel-env

# Activate the environment
pyenv activate wisentinel-env  # Linux/Mac
# or
pyenv shell wisentinel-env    # Windows
```

### 3. Alternative: Using venv

If you prefer not to use pyenv, you can use Python's built-in venv:

```bash
# Create a Python virtual environment
python -m venv wisentinel-env
source wisentinel-env/bin/activate  # Linux/Mac
# or
.\wisentinel-env\Scripts\activate  # Windows
```

### 4. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=src  # Linux/Mac
# or
set PYTHONPATH=src    # Windows
```

### 5. Verify Installation

```bash
# Verify Python version
python --version  # Should show 3.12 or your chosen version

# Verify dependencies
pip list  # Should show all required packages
```

## Overview

WiSentinel provides two main CLI clients:

- **Static Mode** (`cli_static.py`): Processes pre-recorded CSI `.dat` files
- **Live Mode** (`cli_live.py`): Handles real-time CSI data streaming
- **Main Interface** (`cli_interface.py`): Unified entry point for both modes

## 1. Live Mode (`cli_live.py`)

The live CLI tool handles real-time CSI data processing, including:
- Data receiving and sending
- Real-time sanitization
- Feature detection
- Motion detection
- Live visualization

### Basic Usage

```bash
# Using the main interface
python src/cli/cli_interface.py live

# Direct usage - Receive mode with default settings
python src/cli/cli_live.py receive

# Send mode with specific file
python src/cli/cli_live.py send --infile /path/to/csi_file.dat

# Receive with custom host/port
python src/cli/cli_live.py receive \
    --receiver-host 192.168.1.100 \
    --receiver-port 8000
```

### Common Options

```bash
# Enable sanitization
python src/cli/cli_live.py receive --sanitize

# Specify output directory
python src/cli/cli_live.py receive --output-path /path/to/output

# Feature Detection Options
python src/cli/cli_live.py receive \
    --detect-features \     # Enable real-time feature extraction
    --detect-motion \      # Enable motion detection
    --visualize           # Show real-time visualizations

# Combined Example with All Features
python src/cli/cli_live.py receive \
    --sanitize \
    --nonlinear \
    --sto \
    --rco \
    --detect-features \
    --detect-motion \
    --visualize \
    --output-path /path/to/output
```

### Live Visualization

```bash
# Enable real-time plotting
python src/cli/cli_live.py receive --plot

# Configure visualization update interval
python src/cli/cli_live.py receive \
    --plot \
    --update-interval 0.1
```

## 2. Static Mode (`cli_static.py`)

The static CLI tool processes saved CSI data files with options for:
- Batch processing
- Sanitization
- Feature extraction
- Motion detection
- Data visualization

### Basic Commands

```bash
# Using the main interface
python src/cli/cli_interface.py static

# Direct usage - Process a directory of CSI files
python src/cli/cli_static.py process \
    --data-dir /path/to/csi/files \
    --output /path/to/output

# Extract features
python src/cli/cli_static.py extract-features \
    /path/to/csi_file.dat \
    /path/to/output

# Detect motion
python src/cli/cli_static.py detect-motion \
    /path/to/csi_file.dat
```

### Processing Options

```bash
# Enable sanitization with calibration
python src/cli/cli_static.py process \
    --data-dir /path/to/csi/files \
    --sanitize \
    --calib-file /path/to/calibration.dat

# Configure antenna settings
python src/cli/cli_static.py process \
    --data-dir /path/to/csi/files \
    --rx-num 3 \
    --tx-num 3 \
    --tones 56

# Enable logging
python src/cli/cli_static.py process \
    --data-dir /path/to/csi/files \
    --enable-logging
```

### Feature Extraction Options

```bash
# Extract with custom window size
python src/cli/cli_static.py extract-features \
    --window-size 100 \
    /path/to/csi_file.dat \
    /path/to/output

# Specify feature types
python src/cli/cli_static.py extract-features \
    --time-domain \
    --frequency-domain \
    --statistical \
    /path/to/csi_file.dat \
    /path/to/output
```

### Motion Detection Options

```bash
# Adjust detection sensitivity
python src/cli/cli_static.py detect-motion \
    --sensitivity 0.8 \
    /path/to/csi_file.dat

# Custom window configuration
python src/cli/cli_static.py detect-motion \
    --window-size 100 \
    --overlap 0.5 \
    /path/to/csi_file.dat
```

## Output Files and Formats

### CSI Data Files
- Raw CSI data: `.dat` files
- Sanitized data: `.npy` files
- Feature data: `.npz` files

### Visualization Files
- Time domain plots: `*_time_domain.png`
- Frequency domain plots: `*_freq_domain.png`
- Statistical plots: `*_statistical.png`
- PCA plots: `*_pca.png`

### Log Files
- Processing logs: `logs/csi_processing_*.log`
- Error logs: `logs/error_*.log`

## Configuration

Both CLI tools use configuration files located in:
- Live config: `src/cli/config/live_config.json`
- Static config: `src/cli/config/static_config.json`
- Feature config: `src/sanitization/config/feature_config.json`

### Example Configuration

```json
{
    "default_receiver_host": "localhost",
    "default_listen_port": 8000,
    "default_output_path": "output/",
    "visualization": {
        "update_interval": 0.1,
        "plot_style": "seaborn"
    },
    "sanitization": {
        "nonlinear": true,
        "sto": true,
        "rco": true,
        "cfo": false
    }
}
```

## Error Handling

Common error scenarios and solutions:

1. Connection errors:
   ```
   Error: Could not connect to receiver
   Solution: Check network connectivity and port availability
   ```

2. File errors:
   ```
   Error: Invalid CSI file format
   Solution: Ensure file is in correct .dat format
   ```

3. Memory errors:
   ```
   Error: Memory allocation failed
   Solution: Reduce buffer size or process smaller chunks
   ```
```

You will be prompted to choose:
- `1` for Live mode
- `2` for Static file mode

---

## 2. Static File Mode (`cli_static.py`)

This mode processes CSI data from `.dat` files in a specified directory.

**Typical usage:**

```bash
python src/cli/cli_static.py [OPTIONS]
```

### Available Flags

| Flag / Option         | Short | Type     | Description                                              | Default      |
|---------------------- |-------|----------|----------------------------------------------------------|--------------|
| `--data-dir`          | `-d`  | Path     | Directory containing CSI data files                      | Prompted     |
| `--sanitize/--no-sanitize` |       | Bool     | Enable/disable sanitization of CSI data                  | Prompted     |
| `--calib-file`        | `-c`  | Path     | Path to calibration file (for sanitization)              | Prompted     |
| `--output`            | `-o`  | Path     | Output directory for processed data                      | Prompted     |
| `--save-sanitized/--no-save-sanitized` | | Bool | Save sanitized data as NumPy files                       | Prompted     |
| `--enable-logging/--no-logging` | | Bool | Enable logging to file in logs directory                | `False`      |
| `--nonlinear/--no-nonlinear` |   | Bool     | Apply nonlinear sanitization                             | Prompted     |
| `--sto/--no-sto`      |       | Bool     | Apply STO (Sample Time Offset) correction                | Prompted     |
| `--rco/--no-rco`      |       | Bool     | Apply RCO (Radio Chain Offset) correction                | Prompted     |
| `--cfo/--no-cfo`      |       | Bool     | Apply CFO (Carrier Frequency Offset) correction          | Prompted     |
| `--method`            |       | Choice   | Sanitization method: 'selective' (per-packet) or 'batch' (MATLAB-style, all steps in batch) | `selective`  |
| `--template-linear-start` |   | Int      | Start index for template linear interval                 | `20`         |
| `--template-linear-end`   |   | Int      | End index (exclusive) for template linear interval       | `39`         |


**Prompts:**  
If you do not provide a flag, the CLI will prompt you for the value interactively.

**Sanitization Methods:**
- `--method selective` (default): Applies each sanitization step per packet (robust, flexible).
- `--method batch`: Applies all sanitization steps in a single batch (closer to MATLAB pipeline, may be faster for large datasets).

**Template Linear Interval:**
- `--template-linear-start` and `--template-linear-end` control the subcarrier range used for template phase linearity (default: 20 to 39, exclusive).

**Features:**
- You will be prompted to select the order in which `.dat` files are processed.
- Processed and/or sanitized data will be saved as `.npy` files.

---

## 3. Live Mode (`cli_live.py`)

This mode is for real-time CSI data collection and processing.

**Typical usage:**

```bash
python src/cli/cli_live.py [OPTIONS]
```

**Features:**
- Prompts for configuration options and calibration file as needed.
- May require compatible hardware and additional configuration.
- Some options may be set in configuration files (e.g., `live_config.json`).

---

## 4. Logging

The CLI supports detailed logging to help track processing and debug issues:

- Use `--enable-logging` to create a timestamped log file in the `logs` directory
- Log files are named `csi_processing_YYYYMMDD_HHMMSS.log`
- Logs include:
  - Calibration file details
  - CSI data shapes and configurations
  - Processing progress and status
  - Any errors or warnings encountered

Example log file name: `csi_processing_20250713_143022.log`

## 5. Tips

- Always activate your virtual environment before running the CLI.
- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- For help on options, you can run:
  ```bash
  python src/cli/cli_static.py --help
  python src/cli/cli_live.py --help
  ```
- Enable logging with `--enable-logging` when debugging issues or tracking processing progress
