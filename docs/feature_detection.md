# Feature Detection Guide

This guide explains how to use WiSentinel's feature detection capabilities for both real-time and static CSI data analysis.

## Overview

WiSentinel provides robust feature detection capabilities that can:
- Extract time-domain features (energy, variance, peaks)
- Analyze frequency-domain characteristics (PSD, spectral centroid)
- Calculate statistical features (kurtosis, skewness, correlation)
- Detect motion events in real-time
- Visualize features through various plots

## Real-Time Feature Detection

### Using the Live CLI

```bash
# Basic usage with default settings
python src/cli/cli_live.py receive

# Specify output directory for feature data
python src/cli/cli_live.py receive --output-path /path/to/output

# Enable sanitization with feature detection
python src/cli/cli_live.py receive --sanitize --output-path /path/to/output
```

The live feature detector will:
- Process incoming CSI packets in real-time
- Display live visualizations of features
- Alert when motion is detected
- Track feature statistics over time
- Save feature data if output path is specified

### Live Feature Visualization

During live processing, you'll see:
1. Time-domain features:
   - Signal energy over time
   - Variance across subcarriers
   - Detected peaks
2. Frequency-domain features:
   - Power spectral density
   - Spectral centroid
3. Statistical features:
   - Correlation matrices
   - Feature distributions

Motion events are highlighted in real-time with confidence scores.

## Static Feature Detection

### Using the Static CLI

```bash
# Extract features from a single file
python src/cli/cli_static.py extract-features /path/to/csi_file.dat /path/to/output

# Detect motion events in a file
python src/cli/cli_static.py detect-motion /path/to/csi_file.dat

# Process with custom antenna configuration
python src/cli/cli_static.py extract-features \
    --rx-num 3 --tx-num 3 --tones 56 \
    /path/to/csi_file.dat /path/to/output
```

### Output Files

The feature extraction process generates:
1. Feature data files (NPZ format):
   - `{filename}_features.npz` containing all extracted features
2. Visualization plots:
   - `{filename}_time_domain.png`
   - `{filename}_freq_domain.png`
   - `{filename}_statistical.png`
   - `{filename}_pca.png`

### Motion Detection Output

The motion detection command provides:
- Timestamp of detected motion events
- Confidence score for each detection
- Window size used for detection
- Total number of detected events

## Python API Usage

### Real-Time Feature Detection

```python
from sanitization.realtime_features import RealTimeFeatureDetector

# Initialize detector
detector = RealTimeFeatureDetector(
    buffer_size=1000,
    feature_window=100,
    update_interval=0.1,
    enable_visualization=True
)

# Process packets in real-time
for csi_packet in csi_stream:
    update_needed, features = detector.process_packet(csi_packet, timestamp)
    if update_needed:
        # Handle new features
        if features['motion']['detected']:
            print(f"Motion detected! Confidence: {features['motion']['confidence']:.2f}")
```

### Static Feature Detection

```python
from sanitization.feature_detection import CSIFeatureDetector
from sanitization.feature_visualization import FeatureVisualizer

# Initialize components
detector = CSIFeatureDetector()
visualizer = FeatureVisualizer()

# Load and process CSI data
csi_data = load_csi_data(file_path)

# Extract features
features = {
    'time_domain': detector.extract_time_features(csi_data),
    'frequency_domain': detector.extract_frequency_features(csi_data),
    'statistical': detector.extract_statistical_features(csi_data),
    'pca': detector.extract_pca_features(csi_data)
}

# Visualize features
visualizer.plot_all_features(features)
visualizer.save_all_plots(features, output_dir, base_filename)
```

## Feature Details

### Time-Domain Features
- Signal energy: Overall strength of the CSI signal
- Variance: Signal variation across subcarriers
- Peak detection: Identification of significant signal changes

### Frequency-Domain Features
- Power Spectral Density (PSD): Signal power distribution across frequencies
- Spectral centroid: Weighted mean of the frequencies present
- Dominant frequency components: Main frequency components in the signal

### Statistical Features
- Kurtosis: Measure of the signal's "tailedness"
- Skewness: Measure of the signal's asymmetry
- Antenna correlation: Correlation between different antenna pairs

### Motion Detection Features
- Energy-based detection
- Variance thresholding
- Peak analysis
- Confidence scoring

## Configuration

Feature detection parameters can be adjusted through the configuration file:
`src/sanitization/config/feature_config.json`

Key parameters include:
- Detection thresholds
- Window sizes
- Visualization settings
- Motion detection sensitivity
