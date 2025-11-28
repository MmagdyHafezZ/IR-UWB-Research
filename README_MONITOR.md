# IR-UWB Respiration Monitor - Quick Start

## 🚀 How to Run

There is **only ONE file** you need to run:

```bash
python3 monitor.py
```

That's it! This is the unified, production-ready monitor with all features and all fixes.

---

## ✅ What monitor.py Includes

**All Features:**
- ✅ Real-time breathing rate detection
- ✅ Live metrics display (on-demand refresh)
- ✅ Interactive command interface
- ✅ Live visualization (6 plots)
- ✅ Data export to .npz files
- ✅ Hardware and synthetic modes

**All Fixes:**
- ✅ No terminal overwrites
- ✅ Input never disappears
- ✅ Output stays visible
- ✅ Complex type handling fixed
- ✅ Stable, reliable, production-ready

---

## 📖 Usage

### 1. Launch Monitor

```bash
python3 monitor.py
```

### 2. Choose Mode

```
Use hardware? (y/N): n    # Press 'n' for synthetic mode (no hardware needed)
```

### 3. Monitor Starts

You'll see:
```
═══════════════════════════════════════════════════════
    IR-UWB Real-Time Respiration Monitor v2.0
═══════════════════════════════════════════════════════
Unified Production CLI - All Features Included

✓ Monitoring started

Command (h for help): _
```

### 4. Use Commands

| Command | Description |
|---------|-------------|
| `s` or **Enter** | Refresh display |
| `v` | Toggle live visualization (6 plots) |
| `p` | Pause data capture |
| `r` | Resume data capture |
| `e` | Export data to .npz file |
| `x` | Clear buffer |
| `h` | Show help |
| `q` | Quit |

---

## 🎯 Example Session

```bash
# Start monitor
python3 monitor.py

# Choose synthetic mode
Use hardware? (y/N): n

# Monitor shows live status
# Press Enter to refresh display
Command (h for help): [Enter]

# Start visualization
Command (h for help): v
✓ Visualization started

# Let it run for a while...

# Export data
Command (h for help): e
✓ Exported to: output/export_20250127_180045.npz

# Quit
Command (h for help): q
```

---

## 📊 Display Panels

When you refresh (press Enter or type `s`), you'll see:

### Breathing Metrics Panel
- Current breathing rate (BPM)
- Average rate (last 10 measurements)
- Variability (standard deviation)
- Visual bar graph
- Chest range, SNR, quality score

### System Status Panel
- Capture status (running/paused)
- Processing status (ready/processing/error)
- Buffer fill level
- Visualization status

---

## 🔧 Requirements

### Required

```bash
pip3 install rich numpy scipy matplotlib
```

### Optional (for hardware mode)

```bash
# Only needed if you have SDR hardware
pip3 install SoapySDR
```

---

## 📁 File Structure

```
IR-UWB-Research/
├── monitor.py                 ← ⭐ USE THIS (unified CLI)
├── config.py                  ← Configuration
├── processing_fixes.py        ← Signal processing (with fixes)
├── simple_visualization.py    ← Visualization module
├── sdr_capture.py             ← Hardware interface
├── range_time_matrix.py       ← RTM construction
├── preprocessing.py           ← Clutter removal
├── respiration_extraction.py  ← Rate detection
├── vmd.py                     ← VMD decomposition
└── output/                    ← Export directory
    └── export_*.npz           ← Exported data files
```

---

## 🎨 Visualization

Press `v` to open live visualization window with 6 plots:

```
┌─────────────────┬─────────────────┐
│ Rate Trend      │ Waveform        │
├─────────────────┼─────────────────┤
│ SNR Trend       │ Quality Score   │
├─────────────────┼─────────────────┤
│ Chest Detection │ Rate Histogram  │
└─────────────────┴─────────────────┘
```

All plots update in real-time!

**To close visualization:**
- Press `v` again in the CLI, OR
- Close the plot window

---

## 💾 Data Export

Press `e` to export data:

```
✓ Exported to: output/export_20250127_180045.npz
```

**File contains:**
- `raw_data` - Complex IQ radar samples
- `breathing_rates` - Time series of breathing rates
- `snr_values` - SNR measurements
- `quality_scores` - Quality metrics
- `timestamps` - Measurement timestamps

**Load exported data:**

```python
import numpy as np

data = np.load('output/export_20250127_180045.npz')
rates = data['breathing_rates']
snr = data['snr_values']
timestamps = data['timestamps']

print(f"Average breathing rate: {np.mean(rates):.1f} BPM")
```

---

## 🐛 Troubleshooting

### Issue: "No breathing rate detected" (shows 0.0 BPM)

**Solutions:**
1. Wait for buffer to fill (needs 1000+ pulses, shown in status)
2. Check SNR (should be > -10 dB)
3. Press `s` to see detailed status
4. Make sure processing shows "Ready" not "Error"

### Issue: "Visualization doesn't open"

**Solutions:**
1. Close any existing matplotlib windows
2. Check matplotlib backend: `python3 -c "import matplotlib; print(matplotlib.get_backend())"`
3. Press `v` twice (toggle off then on)

### Issue: High CPU usage

**Solution:**
- This is normal when visualization is running
- Press `v` to close visualization when not needed
- Processing uses 20-40% CPU normally

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Buffer size
DATA_BUFFER_SIZE = 5000  # Increase for longer history

# Processing
PROCESSING_INTERVAL = 2.0  # Seconds between processing

# Breathing rate limits
BREATHING_FREQ_MIN = 0.1  # 6 BPM
BREATHING_FREQ_MAX = 0.5  # 30 BPM

# Clutter removal
CLUTTER_METHOD = 'moving_average'  # or 'mean', 'median'
```

---

## 📚 Documentation

For detailed information:
- **QUICK_START.md** - Complete guide
- **SYSTEM_DOCUMENTATION.md** - Technical details
- **HARDWARE_SETUP.md** - SDR hardware setup
- **CLI_GUIDE.md** - Command reference

---

## ✨ Features Highlights

### What Makes This Monitor Great

1. **Stable Input**
   - Command prompt never disappears
   - Type freely without interruption
   - Output stays on screen

2. **Smart Processing**
   - Handles complex IQ data correctly
   - Improved chest detection
   - Detrending and high-pass filtering
   - VMD decomposition support

3. **Real-Time Visualization**
   - 6 live plots in separate window
   - Updates smoothly without crashes
   - Process-safe implementation

4. **Easy to Use**
   - Single command to run
   - Clear status panels
   - Helpful error messages
   - Simple commands

---

## 🎓 Tips & Best Practices

1. **Always export before quitting** if you want to save data
2. **Wait for 1000+ pulses** before expecting accurate rates
3. **Monitor SNR** - values below -10 dB indicate poor signal
4. **Use visualization sparingly** - consumes CPU
5. **Press Enter** to refresh display anytime
6. **Check status** with `s` if something seems wrong

---

## 🚦 Status Indicators

### Capture Status
- "Running (synthetic mode)" - Normal operation
- "Paused" - Capture is paused (press `r` to resume)

### Processing Status
- "Ready" - ✅ Everything working
- "Processing..." - Currently processing
- "Buffering... (500/1000)" - Waiting for more data
- "Error: ..." - ❌ Something wrong (check error message)

### Buffer Level
- Below 20% - Just started
- 20-80% - Normal operation
- Above 80% - Nearly full (old data will be overwritten)

---

## 🎉 Success Metrics

You know it's working when you see:
- ✅ Processing status: "Ready"
- ✅ Current rate: Non-zero (e.g., 15.2 BPM)
- ✅ SNR: Above -10 dB
- ✅ Buffer: Above 1000 pulses
- ✅ No error messages

---

## Summary

**One file to rule them all:**

```bash
python3 monitor.py
```

**Everything works:**
- ✅ Input never disappears
- ✅ All features included
- ✅ All fixes applied
- ✅ Production ready

**Enjoy monitoring!** 🎊

---

**Version:** 2.0 (Unified)
**Date:** 2025-01-27
**Status:** Production Ready ✅
