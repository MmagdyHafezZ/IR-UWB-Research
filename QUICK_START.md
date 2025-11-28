# Quick Start Guide

## 📦 Installation

```bash
# On AIR-T device or development machine
cd ~/IR-UWB-Research
pip3 install numpy scipy matplotlib
```

## 🧪 Test System

```bash
# Run all tests (works without hardware)
python3 tests.py
# Expected: 7/7 tests passed (100%)
```

## 🚀 Run System

### ⭐ Unified Monitor (THE ONLY ONE TO USE!)

```bash
# Launch the unified monitor - ALL features, ALL fixes included
python3 monitor.py
```

**Why monitor.py?**
- ✅ **Only one file** - No confusion about which to use
- ✅ **All features** - Everything from all previous versions
- ✅ **All fixes** - Complex type handling, stable input, no overwrites
- ✅ **Production ready** - Tested and reliable

**Features:**
- **Stable input** - Command prompt never disappears
- **On-demand refresh** - Display updates when you want (press Enter)
- **Live visualization** - Press `v` for 6 live plots
- **Interactive commands** - Full control while monitoring
- **Full threading** - Capture, processing, display in parallel
- **No hardware needed** - Works with realistic synthetic data

**Commands:**
- `s` or **Enter** - Refresh display
- `v` - Toggle live visualization (6 plots)
- `p` - Pause capture
- `r` - Resume capture
- `e` - Export data to .npz file
- `x` - Clear buffer
- `h` - Help
- `q` - Quit

### Classic Pipeline (Batch Processing)

```bash
# Capture and process (batch mode)
python3 main.py --mode realtime
```

### Process Saved Data
```bash
# Offline processing
python3 main.py --mode offline --load output/raw_iq_data.npy
```

### Demonstration Scripts

```bash
# Before/after processing improvements
python3 demo_processing_improvements.py

# Test live visualization only
python3 live_visualization.py
```

## ⚙️ Quick Configuration

Edit `config.py`:

```python
# Change pulse type
IMPULSE_TYPE = "monocycle"  # Options: gaussian, monocycle, doublet, ricker

# Adjust PRF
PULSE_REPETITION_FREQ = 2000  # Hz (higher for better breathing resolution)

# Change capture duration
NUM_PULSES = 5000  # More pulses = more accurate
```

## 📊 Output Files

Results saved to `output/YYYYMMDD_HHMMSS/`:
- `raw_iq_data.npy` - Captured radar data
- `range_time_matrix.png` - 2D heatmap
- `breathing_waveform.png` - Detected breathing
- `analysis_results.json` - Numerical results

## 🔧 Troubleshooting

### "SoapySDR not available"
- **For testing**: This is OK, tests still run
- **For hardware**: Install SoapyAIRT driver on AIR-T

### "No breathing detected"
- Check subject is 1-2 meters from antennas
- Increase `NUM_PULSES` to 2000+
- Try different `IMPULSE_TYPE`

### "Low signal quality"
- Adjust `TX_GAIN` (try -3 dB)
- Ensure clear line of sight to subject's chest
- Minimize environmental clutter

## 📁 Files Overview

| File | Purpose |
|------|---------|
| **Production System** | |
| `monitor.py` | **Unified production CLI (THE ONE TO USE!)** |
| `simple_visualization.py` | **Process-safe live plots** |
| `processing_fixes.py` | **Improved signal processing** |
| `demo_processing_improvements.py` | **Before/after demo** |
| **Core System** | |
| `config.py` | All settings |
| `sdr_capture.py` | Hardware & pulses |
| `range_time_matrix.py` | 2D matrix construction |
| `preprocessing.py` | Signal cleaning |
| `respiration_extraction.py` | Rate detection |
| `vmd.py` | Variational Mode Decomposition |
| `visualization.py` | Plotting functions |
| `main.py` | Classic pipeline runner |
| `tests.py` | Verify system (7 tests) |

## 🔄 Threading Architecture (monitor.py)

The unified monitor uses **3 concurrent threads**:

```
Main Thread (UI)
├── Capture Thread ────► Circular Buffer (thread-safe)
├── Processing Thread ──► Results Queue
│   └── All processing fixes included
└── Visualization ──────► Separate process (when enabled)
```

**Benefits:**
- Non-blocking capture (never misses pulses)
- Parallel processing (CPU-efficient)
- Responsive UI (commands work while processing)
- Live updates (display + plots simultaneously)

**Performance:**
- Capture: 1000 pulses/sec
- Processing: Every 2 seconds
- Display: Auto-refresh every 2 seconds
- Visualization: Updates every 1 second

## 🎯 Typical Workflow

### For New Users (Interactive Mode)

1. **Test without hardware**:
   ```bash
   python3 tests.py
   # Expected: 7/7 passed
   ```

2. **Launch interactive monitor**:
   ```bash
   python3 monitor.py
   # Starts in synthetic mode (no hardware needed)
   ```

3. **Enable live visualization**:
   - Press `v` to see 6 live plots
   - Watch breathing rate update in real-time

4. **Export data when done**:
   - Press `e` to save measurements
   - Press `q` to quit

### For Hardware Users (AIR-T)

1. **Connect AIR-T and antennas**

2. **Run interactive monitor**:
   ```bash
   python3 monitor.py
   # Choose 'y' when asked "Use hardware?"
   ```

3. **Position subject 1-2m from antennas**

4. **Monitor breathing in real-time**

5. **Export and analyze**

## 📈 Expected Performance

- **Range Resolution**: ~4.8 mm
- **Detection Range**: 0.5-5 meters
- **Accuracy**: ±1-2 BPM
- **Update Rate**: Real-time (1 Hz)

---

**Need help?** Check README.md for full documentation