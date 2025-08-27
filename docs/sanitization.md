# Data Sanitization
## Inspiration
The code used for sanitizing the data in this project are based on the MATLAB code in the paper "*Hands-on Wireless Sensing with Wi-Fi: A Tutorial*" by Yang et al. and their colleagues from Tsinghua University. 

This has been readapted using pure Python (NumPy) and does not require converting the `.dat` files from the capturing of data to the `.mat` format of MATLAB. 

## Types of filters
There are 5 main sources of noise that the author mentioned:
- Phase and amplitude errors from nonlinearity.
- Sampling Frequency Offset (SFO) & Packet Detection Delay (PDD) (time offsets).
- Radio Chain Offset (RCO).
- Sampling Time Offset (STO).
- Central Frequency Offset (CFO).

In this case, we have implemented the following:
- STO correction
- RCO correction
- Nonlinearity fixes
- CFO correction

These have been made to also be toggleable (e.g., one or more of the filters can be toggled on or off).

## Why sanitize the data?
The data we get is usually convoluted, with different signals on top of one another, and noise from other sources as well. These can include concrete walls, glass panels, metal items, other people, etc. This is why we want to sanitize the data, and try to denoise the signal as much as possible so that we can make important distinctions for our detection methods.

To go into detail, for each of the above listed error sources, these are the reasons behind such inaccuracies (per the authors of the original paper):
- Nonlinearity in signal: hardware filtering implemented by manufacturers, causing gain to be different between frequency bands

## Using the Sanitization Module

### CLI Usage
The simplest way to use the sanitization functionality is through the CLI clients, where configurations can be set in the `config` folder.

### Programmatic Usage
You can also use the sanitization module directly in your Python code:

```python
from sanitization.sanitize_denoising import SanitizeDenoising, load_csi_data

# Initialize sanitizer with calibration data
sanitizer = SanitizeDenoising(calib_path="dataset/no-person/no_person-1min-1.dat")

# Load your CSI data
csi_data, metadata = load_csi_data("dataset/standing/root/standing-1-0-1.dat")

# Generate template from calibration data
template = sanitizer.set_template(csi_data)

# Apply sanitization with specific filters
sanitized_csi = sanitizer.sanitize_csi_selective(
    csi_data,
    template=template,
    nonlinear=True,  # Apply nonlinear correction
    sto=True,        # Apply Sample Time Offset correction
    rco=True,        # Apply Radio Chain Offset correction
    cfo=False        # Don't apply Carrier Frequency Offset correction
)

# For real-time processing, use RealTimeSanitizeDenoising
from sanitization.sanitize_denoising import RealTimeSanitizeDenoising

# Initialize real-time sanitizer
rt_sanitizer = RealTimeSanitizeDenoising(
    calib_path="dataset/no-person/no_person-1min-1.dat",
    buffer_size=100  # Number of packets to keep in buffer for smoothing
)

# Process packets in real-time
for packet in csi_packets:  # Your packet stream
    sanitized_packet = rt_sanitizer.process_packet(
        packet,
        nonlinear=True,
        sto=True,
        rco=True,
        cfo=False
    )
    # Use sanitized_packet for your application
```

The module provides two main classes:
- `SanitizeDenoising`: For batch processing of CSI data
- `RealTimeSanitizeDenoising`: For streaming CSI data with buffer management

## Calibration Data Requirement
This is data that we have captured using two routers connected to one another using Ethernet cables and a device which has SSH control over the routers running OpenWRT, whose specific firmware can be checked out [here](docs/hardware_setup.md). Besides these elements, there is no one in the room (no human presence), which helps us create a picture in the signaling of how the room and its elements interact with the signal.

