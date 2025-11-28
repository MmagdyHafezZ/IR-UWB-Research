# IR-UWB Respiration Detection System

Real-time breathing rate (BPM) detection using SoapySDR and IR-UWB radar.

## Setup

### 1. Install Requirements

```bash
pip3 install numpy scipy matplotlib rich SoapySDR
```

### 2. Connect SDR Hardware

- Connect your SDR device (AIR-T or compatible)
- Connect UWB antennas to TX/RX ports
- Position antennas facing the monitoring area

## Run

```bash
python3 monitor.py
```

When prompted:
- **Use hardware? (y/N):** Press `y` if SDR is connected, `n` for demo mode

## See Live BPM

Once running, you'll see:

```
╭─── Breathing Metrics ──────────────────╮
│ Current Rate    15.2 BPM              │
│ Average Rate    15.0 BPM              │
│ Variability     0.3 BPM               │
╰────────────────────────────────────────╯
```

### Commands

- **Enter** - Refresh BPM display
- **v** - Open live visualization window (shows breathing waveform)
- **q** - Quit

### Positioning

- Subject should be **0.5 to 3 meters** from antennas
- Point antennas at chest area
- Minimize movement for best results

## Configuration

Edit `config.py` to adjust:

```python
PULSE_REPETITION_FREQ = 2000  # Hz (sampling rate)
TX_GAIN = -3  # dB (transmit power)
CENTER_FREQ = 4.3e9  # Hz (4.3 GHz)
```

---

**That's it.** The system will detect breathing and display BPM in real-time.