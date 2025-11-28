# Quick Start - IR-UWB BPM Detection

## Install

```bash
pip3 install numpy scipy matplotlib rich SoapySDR
```

## Run

```bash
python3 monitor.py
```

Choose:
- **y** = Use SDR hardware
- **n** = Demo mode (no hardware)

## View BPM

The screen shows live breathing rate:

```
Current Rate    15.2 BPM
Average Rate    15.0 BPM
Variability     0.3 BPM
```

Press **Enter** to refresh.
Press **v** for live waveform.
Press **q** to quit.

## Hardware Setup

1. Connect SDR device
2. Attach UWB antennas
3. Position 0.5-3m from subject
4. Point at chest

That's all you need to detect and see BPM.