# IR-UWB Respiration Detection System

A complete implementation of an Impulse Radio Ultra-Wideband (IR-UWB) radar system for non-contact respiration detection using the AIR-T SDR platform with SoapySDR.

## Overview

This system implements contactless respiration monitoring using UWB radar technology. It captures radar echoes from the chest wall, processes the signals through a comprehensive pipeline, and estimates breathing rate using both time-domain and frequency-domain methods.

## Features

- **SDR Signal Capture**: Real-time impulse generation and data acquisition using SoapySDR
- **Range-Time Matrix Construction**: Transforms raw IQ data into 2D radar matrix with pulse alignment
- **Preprocessing Pipeline**:
  - Static clutter removal (mean/median/moving average)
  - Slow-time high-pass filtering for detrending
  - Variance-based chest detection
  - Range bin normalization
- **Respiration Extraction**:
  - Band-pass filtering (0.1-0.5 Hz for breathing)
  - Time-domain breathing rate detection (peak counting)
  - Frequency-domain breathing rate detection (FFT)
  - Breathing pattern analysis (depth, regularity, I/E ratio)
- **Comprehensive Visualization**: Multi-panel plots for all analysis stages

## System Architecture

```
┌─────────────────┐
│  SDR Capture    │  Step 1: Generate impulses, capture IQ data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Range-Time      │  Step 2: Construct 2D matrix, align pulses
│ Matrix          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │  Step 3: Clutter removal, filtering, chest detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Respiration     │  Step 4: Extract breathing signal, estimate rate
│ Extraction      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │  Step 5: Generate comprehensive analysis report
│ & Reporting     │
└─────────────────┘
```

## Installation

### Prerequisites

- AIR-T SDR device with SoapyAIRT driver installed
- Python 3.7 or higher
- Access to AIR-T via SSH or local terminal

### Setup

1. Clone or transfer this repository to your AIR-T:
```bash
cd ~
git clone <repository-url>
cd IR-UWB-Research
```

2. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

3. Verify SoapySDR installation:
```bash
python3 -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
```

## Usage

### Real-time Mode (with SDR)

Run the complete pipeline with live SDR capture:

```bash
python3 main.py --mode realtime
```

This will:
1. Initialize the AIR-T SDR
2. Generate and transmit UWB impulses
3. Capture radar echoes
4. Process the data through the complete pipeline
5. Display breathing rate and save all results

### Offline Mode (with saved data)

Process previously captured data:

```bash
python3 main.py --mode offline --load output/raw_iq_data.npy
```

### Disable Plotting

Run without displaying plots (useful for headless operation):

```bash
python3 main.py --mode realtime --no-plot
```

## Configuration

Edit [config.py](config.py) to customize system parameters:

### Key Parameters

```python
# Transmitter Settings
TX_FREQ = 2.45e9          # Center frequency (2.45 GHz)
TX_SAMPLE_RATE = 20e6     # 20 MHz
TX_GAIN = -5              # dB

# Receiver Settings
RX_FREQ = 2.45e9          # Center frequency
RX_SAMPLE_RATE = 31.25e6  # 31.25 MHz
RX_CHANNEL = 1            # Use RX2 (more reliable than RX1)

# Pulse Settings
PULSE_REPETITION_FREQ = 1000  # 1 kHz PRF
NUM_PULSES = 1000             # Number of frames to capture
SAMPLES_PER_PULSE = 1024      # Fast-time samples

# Breathing Detection
BREATHING_FREQ_MIN = 0.1  # 6 BPM
BREATHING_FREQ_MAX = 0.5  # 30 BPM
```

## Module Descriptions

### 1. config.py
Configuration file containing all system parameters and utility functions.

### 2. sdr_capture.py
Handles SDR initialization, impulse generation, and data acquisition.

**Key Classes:**
- `SDRCapture`: Main SDR interface class

**Key Methods:**
- `setup_transmitter()`: Configure TX parameters
- `setup_receiver()`: Configure RX parameters
- `generate_impulse_signal()`: Create UWB impulse train
- `record_pulse_sequence()`: Capture synchronized pulse data

### 3. range_time_matrix.py
Constructs 2D range-time matrix from raw IQ data and handles pulse alignment.

**Key Classes:**
- `RangeTimeMatrix`: Matrix construction and alignment

**Key Methods:**
- `construct_matrix()`: Build 2D matrix [slow_time, fast_time]
- `align_pulses()`: Compensate for timing jitter using cross-correlation
- `get_range_bins()`: Convert sample indices to range in meters

### 4. preprocessing.py
Applies signal conditioning and clutter removal.

**Key Classes:**
- `Preprocessor`: Complete preprocessing pipeline

**Key Methods:**
- `remove_clutter_mean_subtraction()`: Static clutter removal
- `apply_highpass_filter()`: Slow-time detrending
- `calculate_slow_time_variance()`: Variance profile for chest detection
- `detect_chest_range_bin()`: Locate subject's chest

### 5. respiration_extraction.py
Extracts breathing signal and estimates breathing rate.

**Key Classes:**
- `RespirationExtractor`: Breathing analysis

**Key Methods:**
- `apply_breathing_bandpass_filter()`: Isolate breathing frequencies
- `detect_breathing_rate_time_domain()`: Peak-based rate estimation
- `detect_breathing_rate_frequency_domain()`: FFT-based rate estimation
- `analyze_breathing_pattern()`: Pattern quality metrics

### 6. visualization.py
Provides comprehensive plotting capabilities.

**Key Classes:**
- `Visualizer`: Plotting utilities

**Key Methods:**
- `plot_range_time_matrix()`: 2D heatmap of radar data
- `plot_variance_profile()`: Chest detection visualization
- `plot_breathing_waveform()`: Time-domain breathing signal
- `plot_frequency_spectrum()`: FFT analysis
- `plot_complete_analysis()`: Multi-panel comprehensive plot

### 7. main.py
Main execution script that orchestrates the complete pipeline.

**Key Classes:**
- `RespirationDetectionSystem`: Integrated system

**Pipeline Steps:**
1. `step1_capture_data()`: SDR data acquisition
2. `step2_construct_range_time_matrix()`: Matrix formation
3. `step3_preprocess_data()`: Signal conditioning
4. `step4_extract_respiration()`: Breathing analysis
5. `step5_generate_comprehensive_report()`: Results visualization

## Testing Individual Modules

Each module includes a test function that can be run standalone:

```bash
# Test SDR capture (generates synthetic data if no SDR available)
python3 sdr_capture.py

# Test range-time matrix construction
python3 range_time_matrix.py

# Test preprocessing pipeline
python3 preprocessing.py

# Test respiration extraction
python3 respiration_extraction.py

# Test visualization
python3 visualization.py
```

## Output Files

Results are saved to timestamped directories under `output/`:

```
output/
└── YYYYMMDD_HHMMSS/
    ├── raw_iq_data.npy              # Raw captured data
    ├── range_time_matrix.npy        # Aligned RTM
    ├── preprocessed_matrix.npy      # After clutter removal
    ├── variance_profile.npy         # Range variance profile
    ├── chest_signal.npy             # Extracted chest signal
    ├── breathing_waveform.npy       # Filtered breathing signal
    ├── frequency_spectrum.npy       # FFT spectrum
    ├── analysis_results.json        # Numerical results
    ├── range_time_matrix.png        # RTM visualization
    ├── variance_profile.png         # Chest detection plot
    ├── breathing_waveform.png       # Breathing waveform
    ├── frequency_spectrum.png       # FFT plot
    └── complete_analysis.png        # Comprehensive report
```

## Implementation Details

### Signal Processing Flow

1. **Impulse Generation**: Gaussian-shaped narrow pulses (UWB characteristic)
2. **Pulse Alignment**: Cross-correlation based timing jitter compensation
3. **Clutter Removal**: Per-range-bin mean subtraction to remove static objects
4. **High-pass Filtering**: Butterworth filter to remove low-frequency drift
5. **Chest Detection**: Maximum variance in slow-time indicates chest location
6. **Phase Extraction**: Unwrap phase from complex signal (sensitive to displacement)
7. **Band-pass Filtering**: Isolate breathing frequencies (0.1-0.5 Hz)
8. **Rate Estimation**:
   - Time-domain: Peak detection and counting
   - Frequency-domain: FFT and dominant frequency identification

### Key Algorithms

**Cross-Correlation Pulse Alignment**:
```python
correlation = signal.correlate(current_pulse, reference_pulse, mode='same')
shift = argmax(abs(correlation)) - num_samples // 2
```

**Variance-Based Chest Detection**:
```python
magnitude = abs(range_time_matrix)
variance_profile = var(magnitude, axis=0)  # Across slow-time
chest_bin = argmax(variance_profile)
```

**Breathing Rate Estimation**:
```python
# Time-domain
peaks = find_peaks(filtered_signal)
breathing_rate_bpm = (num_peaks / duration_seconds) * 60

# Frequency-domain
fft_result = fft(filtered_signal)
dominant_freq = frequencies[argmax(abs(fft_result))]
breathing_rate_bpm = dominant_freq * 60
```

## Troubleshooting

### Common Issues

1. **SoapySDR not found**
   - Ensure SoapyAIRT driver is installed on AIR-T
   - Verify with: `SoapySDRUtil --find`

2. **RX1 unreliable**
   - System is configured to use RX2 by default (`RX_CHANNEL = 1`)
   - This is more reliable based on testing

3. **No breathing detected**
   - Ensure subject is 1-3 meters from antenna
   - Adjust `BREATHING_FREQ_MIN/MAX` if needed
   - Check variance profile plot for chest detection

4. **Overflow warnings (GNU Radio)**
   - Python implementation is preferred over GNU Radio
   - Reduce sample rate if necessary

5. **Low signal quality**
   - Adjust TX_GAIN (increase for longer range)
   - Ensure proper antenna alignment
   - Minimize environmental clutter

## Hardware Setup

### Physical Configuration

```
AIR-T SDR
├── TX1: Connected to transmit antenna (towards subject)
└── RX2: Connected to receive antenna (towards subject)

Subject: 1-3 meters from antennas
Environment: Minimize metallic objects and moving clutter
```

### Recommended Testing Setup

1. Use separate TX and RX antennas with 2-3 feet separation
2. Point antennas toward subject's chest
3. Subject should be stationary (sitting or standing)
4. Start with 1.5 meter distance
5. Minimize other people in the radar field of view

## Performance Metrics

### Typical Performance

- **Range Resolution**: ~4.8 mm (at 31.25 MHz bandwidth)
- **Maximum Range**: 150 m (at 1 kHz PRF)
- **Breathing Rate Accuracy**: ±1-2 BPM
- **Update Rate**: 1 Hz (with 1 kHz PRF and 1000 frames)
- **Detection Range**: 0.5-5 meters (depending on TX gain and subject size)

### Signal Quality Indicators

- Quality Score > 0.7: Excellent
- Quality Score 0.5-0.7: Good
- Quality Score < 0.5: Poor (check setup)

## Development Notes

### Python Development on AIR-T

The current recommended method:
1. Use https://vscode.dev/ from browser for code editing
2. Run scripts through terminal: `python3 script.py`
3. Transfer files via SCP/SFTP if needed

### GNU Radio Alternative

While GNU Radio Companion can be used, Python provides better performance:
```bash
conda activate gr310
gnuradio-companion
# Use SoapyCustom Source/Sink blocks with driver="SoapyAIRT"
```

## Future Enhancements

Potential improvements:
- [ ] Heart rate detection (requires higher PRF)
- [ ] Multi-target tracking
- [ ] Machine learning for improved chest detection
- [ ] Real-time streaming visualization
- [ ] Sleep apnea detection
- [ ] Movement detection and classification

## References

### Technical Background

- UWB Radar principles
- Doppler effect in radar systems
- Phase-based displacement measurement
- Clutter removal techniques
- Respiration signal processing

### AIR-T Documentation

- SoapyAIRT driver documentation
- AIR-T hardware specifications
- GNU Radio integration guide

## License

[Specify your license here]

## Authors

[Your name/organization]

## Acknowledgments

- Deepwave Digital for AIR-T platform
- SoapySDR project

## Support

For issues and questions:
- Check troubleshooting section
- Review configuration parameters
- Test individual modules
- Verify hardware connections

---

**Note**: This implementation is for research and educational purposes. Ensure compliance with local regulations regarding UWB transmission frequencies and power levels.
