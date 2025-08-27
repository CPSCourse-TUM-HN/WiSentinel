# CLI Usage Guide

This guide covers how to use WiSentinel's command-line interface (CLI) tools for both live and static CSI data processing.

## Installation

Before using the CLI tools, ensure you have all required dependencies installed:

```bash
# Create a Python virtual environment (recommended)
python -m venv wisentinel-env
source wisentinel-env/bin/activate  # Linux/Mac
# or
.\wisentinel-env\Scripts\activate  # Windows

# Install required packages
pip install -r requirements.txt
```

## Live CLI (`cli_live.py`)

The live CLI tool handles real-time CSI data processing, including:
- Data receiving and sending
- Real-time sanitization
- Feature detection
- Motion detection
- Live visualization

### Basic Usage

```bash
# Receive mode with default settings
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

# Enable feature detection
python src/cli/cli_live.py receive --detect-features

# Customize sanitization
python src/cli/cli_live.py receive \
    --sanitize \
    --nonlinear \
    --sto \
    --rco \
    --cfo
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

## Static CLI (`cli_static.py`)

The static CLI tool processes saved CSI data files with options for:
- Batch processing
- Sanitization
- Feature extraction
- Motion detection
- Data visualization

### Basic Commands

```bash
# Process a directory of CSI files
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
