"""
Configuration file for IR-UWB Respiration Detection System
"""

import numpy as np

class Config:
    """Configuration parameters for the IR-UWB system"""

    
    SDR_DRIVER = "SoapyAIRT"

    
    TX_CHANNEL = 0  
    TX_FREQ = 4.3e9  
    TX_SAMPLE_RATE = 20e6  
    TX_GAIN = 0  # Slightly higher TX gain for better SNR (adjust if saturation occurs)

    
    RX_CHANNEL = 0  # Stay on RX1 as requested
    RX_FREQ = 4.3e9  
    RX_SAMPLE_RATE = 31.25e6  
    RX_GAIN_MODE = False  # Use manual gain for stability
    RX_GAIN_DB = 35  # Manual RX gain in dB (tune down if clipping, up if too noisy)
    RX_BITS = 16  
    RX_TIMEOUT_US = int(5e6)  

    
    IMPULSE_TYPE = "gaussian"  
    IMPULSE_WIDTH = 10  
    IMPULSE_AMPLITUDE = 0.5  
    PULSE_REPETITION_FREQ = 1000  

    
    NUM_PULSES = 20000  # ~20 seconds at 1 kHz PRF for reliable breathing detection
    SAMPLES_PER_PULSE = 1024  
    RECORDING_DURATION = 30  # seconds; avoid unbounded captures by default

    
    RANGE_BINS = SAMPLES_PER_PULSE  
    SLOW_TIME_FRAMES = NUM_PULSES  

    
    C = 3e8  

    
    CLUTTER_REMOVAL_METHOD = "mean_subtraction"  
    HIGHPASS_CUTOFF = 0.05  
    VARIANCE_THRESHOLD = 0.1  

    
    BREATHING_FREQ_MIN = 0.1  
    BREATHING_FREQ_MAX = 0.5  
    FILTER_ORDER = 4  

    
    USE_VMD = True  
    VMD_NUM_MODES = 4  
    VMD_ALPHA = 2000  
    VMD_TAU = 0.0  
    VMD_DC_PART = False  
    VMD_INIT_METHOD = 1  
    VMD_TOL = 1e-7  
    VMD_MAX_ITER = 500  

    
    SAVE_RAW_DATA = True
    SAVE_PROCESSED_DATA = True
    OUTPUT_DIR = "output"
    PLOT_RESULTS = True

    @staticmethod
    def get_range_resolution():
        """Calculate range resolution based on bandwidth"""
        bandwidth = Config.RX_SAMPLE_RATE
        return Config.C / (2 * bandwidth)

    @staticmethod
    def get_max_range():
        """Calculate maximum unambiguous range"""
        prf = Config.PULSE_REPETITION_FREQ
        return Config.C / (2 * prf)

    @staticmethod
    def get_pulse_interval():
        """Get time interval between pulses"""
        return 1.0 / Config.PULSE_REPETITION_FREQ

    @staticmethod
    def validate_parameters():
        """Validate configuration parameters"""
        errors = []
        warnings = []

        
        nyquist = 2 * Config.BREATHING_FREQ_MAX
        if Config.PULSE_REPETITION_FREQ < nyquist:
            errors.append(f"PRF {Config.PULSE_REPETITION_FREQ} Hz < Nyquist {nyquist} Hz")

        
        min_duration = 2.0 / Config.BREATHING_FREQ_MIN
        actual_duration = Config.NUM_PULSES / Config.PULSE_REPETITION_FREQ
        if actual_duration < min_duration:
            warnings.append(f"Capture duration {actual_duration:.1f}s < recommended {min_duration:.1f}s")

        
        range_res = Config.get_range_resolution()
        if range_res > 0.01:  
            warnings.append(f"Range resolution {range_res*1000:.1f}mm may be coarse")

        return errors, warnings

    @staticmethod
    def print_config():
        """Print key configuration parameters"""
        print("=" * 60)
        print("IR-UWB Respiration Detection System Configuration")
        print("=" * 60)
        print(f"TX Frequency: {Config.TX_FREQ/1e9:.2f} GHz")
        print(f"RX Frequency: {Config.RX_FREQ/1e9:.2f} GHz")
        print(f"TX Sample Rate: {Config.TX_SAMPLE_RATE/1e6:.2f} MHz")
        print(f"RX Sample Rate: {Config.RX_SAMPLE_RATE/1e6:.2f} MHz")
        print(f"Pulse Repetition Frequency: {Config.PULSE_REPETITION_FREQ} Hz")
        print(f"Pulse Interval: {Config.get_pulse_interval()*1e3:.2f} ms")
        print(f"Number of Pulses: {Config.NUM_PULSES}")
        print(f"Samples per Pulse: {Config.SAMPLES_PER_PULSE}")
        print(f"Range Resolution: {Config.get_range_resolution():.4f} m")
        print(f"Maximum Range: {Config.get_max_range():.2f} m")
        print(f"Breathing Frequency Range: {Config.BREATHING_FREQ_MIN}-{Config.BREATHING_FREQ_MAX} Hz")

        
        errors, warnings = Config.validate_parameters()
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ! {warning}")
        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  ✗ {error}")

        print("=" * 60)
