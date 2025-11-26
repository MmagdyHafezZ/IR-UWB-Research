# Quick Start Guide

## 📦 Installation

```bash
# On AIR-T device
cd ~/IR-UWB-Research
pip3 install numpy scipy matplotlib
```

## 🧪 Test System

```bash
# Run all tests (works without hardware)
python3 tests.py
# Expected: 6/6 tests passed
```

## 🚀 Run System

### With AIR-T Hardware
```bash
# Real-time capture and processing
python3 main.py --mode realtime
```

### Process Saved Data
```bash
# Offline processing
python3 main.py --mode offline --load output/raw_iq_data.npy
```

### Without Plots (headless)
```bash
python3 main.py --mode realtime --no-plot
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
| `config.py` | All settings |
| `sdr_capture.py` | Hardware & pulses |
| `range_time_matrix.py` | 2D matrix |
| `preprocessing.py` | Signal cleaning |
| `respiration_extraction.py` | Rate detection |
| `visualization.py` | Plotting |
| `main.py` | Run everything |
| `tests.py` | Verify system |

## 🎯 Typical Workflow

1. **Test without hardware**:
   ```bash
   python3 tests.py
   ```

2. **Connect AIR-T and antennas**

3. **Run real-time**:
   ```bash
   python3 main.py --mode realtime
   ```

4. **Check results** in `output/` directory

5. **Adjust config** if needed and repeat

## 📈 Expected Performance

- **Range Resolution**: ~4.8 mm
- **Detection Range**: 0.5-5 meters
- **Accuracy**: ±1-2 BPM
- **Update Rate**: Real-time (1 Hz)

---

**Need help?** Check README.md for full documentation