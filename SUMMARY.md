# IR-UWB Respiration Detection System - Consolidated Implementation

## ✅ Clean, Production-Ready Codebase

All "improved" and "optimized" files have been consolidated. The system now has a clean structure with all related functions in their proper modules.

## 📁 Final File Structure

```
IR-UWB-Research/
├── config.py                    # System configuration with validation
├── sdr_capture.py               # SDR + pulse generation functions
├── range_time_matrix.py         # 2D matrix construction & alignment
├── preprocessing.py             # Clutter removal & signal conditioning
├── respiration_extraction.py    # Breathing rate detection (time & freq domain)
├── visualization.py             # Plotting and visualization
├── main.py                      # Complete pipeline orchestration
├── tests.py                     # Consolidated test suite (6 tests)
└── README.md                    # Complete documentation
```

**Total: 9 clean files** (no duplicates, no "improved" variants)

## 🔧 Consolidated Features

### config.py
- All configuration parameters
- **Built-in validation** (checks PRF, resolution, duration)
- Pulse type selection: gaussian, monocycle, doublet, ricker

### sdr_capture.py
- **4 pulse generation functions** (consolidated from separate module):
  - `generate_gaussian_pulse()` - Simple Gaussian
  - `generate_gaussian_monocycle()` - Zero DC component
  - `generate_gaussian_doublet()` - Better UWB spectrum
  - `generate_ricker_wavelet()` - Mexican hat wavelet
- SDR hardware interface (SoapySDR)
- Synchronized TX/RX operation

### tests.py
- **6 comprehensive tests** (consolidated from 3 separate test files):
  1. Configuration validation
  2. Pulse generation (all 4 types)
  3. Range-time matrix construction
  4. Preprocessing pipeline
  5. Respiration extraction
  6. Complete end-to-end pipeline
- **100% test pass rate**

## 🎯 Usage

### Run Tests
```bash
python3 tests.py
# Output: 6/6 tests passed (100%)
```

### Real-Time Operation (with AIR-T)
```bash
python3 main.py --mode realtime
```

### Process Saved Data
```bash
python3 main.py --mode offline --load output/raw_iq_data.npy
```

## ⚙️ Key Improvements from Consolidation

1. **No Redundancy**: Removed improved_impulse_generator.py, optimized_config.py
2. **Functions in Proper Place**: Pulse generation is in sdr_capture.py where it belongs
3. **Single Test Suite**: Consolidated test_suite.py, test_system.py, run_basic_test.py → tests.py
4. **Optional SoapySDR**: Tests run without hardware (graceful degradation)
5. **Validation Built-in**: Config automatically validates parameters

## 🧪 Test Results

```
✓ PASS: Configuration
✓ PASS: Pulse Generation
✓ PASS: Range-Time Matrix
✓ PASS: Preprocessing
✓ PASS: Respiration Extraction
✓ PASS: Complete Pipeline

Results: 6/6 tests passed (100%)
🎉 ALL TESTS PASSED!
```

## 📊 System Specifications

- **Range Resolution**: 4.8 mm
- **Maximum Range**: 150 m
- **Breathing Detection**: 6-30 BPM (0.1-0.5 Hz)
- **Pulse Types**: 4 options (gaussian, monocycle, doublet, ricker)
- **Detection Methods**: Time-domain + Frequency-domain

## 🚀 Ready for Deployment

The system is fully functional and ready for AIR-T hardware integration:

1. **Clean codebase** - no redundant files
2. **Tested components** - 100% test pass rate
3. **Hardware optional** - tests run without SoapySDR
4. **Well documented** - complete README included
5. **Professional structure** - all functions in logical modules

## 📝 Configuration Example

```python
# In config.py
IMPULSE_TYPE = "gaussian"         # or "monocycle", "doublet", "ricker"
IMPULSE_WIDTH = 10                # samples
PULSE_REPETITION_FREQ = 1000     # Hz
RX_CHANNEL = 1                    # RX2 (more reliable)
```

## 🔄 Processing Pipeline

```
1. Generate UWB impulse → sdr_capture.py
2. Construct range-time matrix → range_time_matrix.py
3. Remove clutter, detect chest → preprocessing.py
4. Extract breathing, estimate rate → respiration_extraction.py
5. Visualize results → visualization.py
```

---

**Implementation Status**: ✅ Complete and Production-Ready

All files are clean, consolidated, and tested. No "improved" or "optimized" duplicates remain.