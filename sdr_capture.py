"""
SDR Signal Capture Module for IR-UWB System
Handles impulse generation, transmission, and reception using SoapySDR
"""

import time
from config import Config
import numpy as np
from scipy import signal as scipy_signal
if not hasattr(scipy_signal, "find_peaks"):
    def _fallback_find_peaks(x, height=None, distance=None, prominence=None):
        """
        Simple local-maximum peak finder as a fallback for old SciPy.
        Only supports a subset of the real find_peaks options.
        """
        x = np.asarray(x)
        # basic local maxima
        peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1

        # Optional height filtering
        if height is not None:
            if isinstance(height, dict):
                hmin = height.get("min", None)
            else:
                hmin = height
            if hmin is not None:
                peaks = peaks[x[peaks] >= hmin]

        properties = {}
        if height is not None:
            properties["peak_heights"] = x[peaks]

        # NOTE: distance/prominence are ignored in this simple fallback
        return peaks, properties

    scipy_signal.find_peaks = _fallback_find_peaks

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_CS16
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    print("Warning: SoapySDR not available - SDR hardware functions will not work")
    
    SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_CS16 = 0, 1, 2, 3



def generate_gaussian_monocycle(width_samples, amplitude=0.5):
    """Generate Gaussian monocycle (first derivative) - zero DC component"""
    if width_samples < 3:
        width_samples = 3
    t = np.arange(width_samples)
    center = width_samples / 2
    sigma = width_samples / 6
    monocycle = -(t - center) / (sigma**2) * np.exp(-((t - center)**2) / (2 * sigma**2))
    if np.max(np.abs(monocycle)) > 0:
        monocycle = monocycle / np.max(np.abs(monocycle))
    return (amplitude * monocycle).astype(np.complex64)


def generate_gaussian_doublet(width_samples, amplitude=0.5):
    """Generate Gaussian doublet (second derivative) - better UWB spectral properties"""
    if width_samples < 5:
        width_samples = 5
    t = np.arange(width_samples)
    center = width_samples / 2
    sigma = width_samples / 6
    term1 = 1 - ((t - center)**2) / (sigma**2)
    term2 = np.exp(-((t - center)**2) / (2 * sigma**2))
    doublet = term1 * term2 / (sigma**2)
    if np.max(np.abs(doublet)) > 0:
        doublet = doublet / np.max(np.abs(doublet))
    return (amplitude * doublet).astype(np.complex64)


def generate_ricker_wavelet(width_samples, amplitude=0.5):
    """
    Generate Ricker wavelet (Mexican hat) - common in UWB radar

    The Ricker wavelet is the second derivative of a Gaussian:
    ψ(t) = (1 - (t/σ)²) * exp(-(t/σ)²/2)
    """
    if width_samples < 10:
        width_samples = 10

    
    t = np.arange(width_samples) - width_samples // 2

    
    sigma = width_samples / 8.0

    
    t_norm = t / sigma
    wavelet = (1 - t_norm**2) * np.exp(-t_norm**2 / 2)

    
    if np.max(np.abs(wavelet)) > 0:
        wavelet = wavelet / np.max(np.abs(wavelet))

    return (amplitude * wavelet).astype(np.complex64)


def generate_gaussian_pulse(width_samples, amplitude=0.5):
    """Generate simple Gaussian pulse"""
    if width_samples < 3:
        width_samples = 3
    t = np.arange(width_samples)
    center = width_samples / 2
    sigma = width_samples / 6
    gaussian = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    return (amplitude * gaussian).astype(np.complex64)


class SDRCapture:
    """Handles SDR initialization and data capture"""

    def __init__(self, hardware_mode=True):
        """
        Initialize SDR device

        Args:
            hardware_mode: If True, initialize hardware immediately. If False, defer for offline/testing.
        """
        self.sdr = None
        self.tx_stream = None
        self.rx_stream = None
        self.impulse_signal = None
        self.hardware_mode = hardware_mode
        self.streams_active = False

        if hardware_mode:
            if not SOAPY_AVAILABLE:
                raise RuntimeError("SoapySDR is not available. Install SoapySDR for hardware operation.")

            self._initialize_hardware()

    def _initialize_hardware(self):
        """Initialize hardware connection (internal method)"""
        if not SOAPY_AVAILABLE:
            raise RuntimeError("SoapySDR is not available. Install SoapySDR for hardware operation.")

        print("Initializing SDR device...")
        args = dict(driver=Config.SDR_DRIVER)
        self.sdr = SoapySDR.Device(args)

        
        print("Available devices:")
        print(SoapySDR.Device.enumerate())

    def setup_transmitter(self):
        """Configure transmitter parameters"""
        if self.sdr is None:
            raise RuntimeError("Hardware not initialized. Cannot setup transmitter.")

        print(f"Setting up transmitter on TX channel {Config.TX_CHANNEL}...")

        
        self.sdr.setSampleRate(SOAPY_SDR_TX, Config.TX_CHANNEL, Config.TX_SAMPLE_RATE)
        actual_rate = self.sdr.getSampleRate(SOAPY_SDR_TX, Config.TX_CHANNEL)
        print(f"TX Sample Rate: {actual_rate/1e6:.2f} MHz")

        
        self.sdr.setFrequency(SOAPY_SDR_TX, Config.TX_CHANNEL, Config.TX_FREQ)
        actual_freq = self.sdr.getFrequency(SOAPY_SDR_TX, Config.TX_CHANNEL)
        print(f"TX Frequency: {actual_freq/1e9:.4f} GHz")

        
        self.sdr.setGain(SOAPY_SDR_TX, Config.TX_CHANNEL, Config.TX_GAIN)
        actual_gain = self.sdr.getGain(SOAPY_SDR_TX, Config.TX_CHANNEL)
        print(f"TX Gain: {actual_gain} dB")

        
        self.tx_stream = self.sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
        print("Transmitter setup complete.")

    def setup_receiver(self):
        """Configure receiver parameters"""
        if self.sdr is None:
            raise RuntimeError("Hardware not initialized. Cannot setup receiver.")

        print(f"Setting up receiver on RX channel {Config.RX_CHANNEL}...")

        
        self.sdr.setSampleRate(SOAPY_SDR_RX, Config.RX_CHANNEL, Config.RX_SAMPLE_RATE)
        actual_rate = self.sdr.getSampleRate(SOAPY_SDR_RX, Config.RX_CHANNEL)
        print(f"RX Sample Rate: {actual_rate/1e6:.2f} MHz")

        
        self.sdr.setGainMode(SOAPY_SDR_RX, Config.RX_CHANNEL, Config.RX_GAIN_MODE)
        print(f"RX AGC Enabled: {Config.RX_GAIN_MODE}")

        
        self.sdr.setFrequency(SOAPY_SDR_RX, Config.RX_CHANNEL, Config.RX_FREQ)
        actual_freq = self.sdr.getFrequency(SOAPY_SDR_RX, Config.RX_CHANNEL)
        print(f"RX Frequency: {actual_freq/1e9:.4f} GHz")

        
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [Config.RX_CHANNEL])
        print("Receiver setup complete.")

    def generate_impulse_signal(self):
        """
        Generate impulse-like excitation signal
        Creates a narrow pulse repeated at the PRF
        """
        print(f"Generating impulse signal (type: {Config.IMPULSE_TYPE})...")

        
        samples_per_interval = int(Config.TX_SAMPLE_RATE / Config.PULSE_REPETITION_FREQ)

        
        impulse = np.zeros(samples_per_interval, dtype=np.complex64)

        
        pulse_generators = {
            'gaussian': generate_gaussian_pulse,
            'monocycle': generate_gaussian_monocycle,
            'doublet': generate_gaussian_doublet,
            'ricker': generate_ricker_wavelet
        }

        if Config.IMPULSE_TYPE in pulse_generators:
            single_pulse = pulse_generators[Config.IMPULSE_TYPE](
                Config.IMPULSE_WIDTH, Config.IMPULSE_AMPLITUDE
            )
        else:
            print(f"Unknown pulse type '{Config.IMPULSE_TYPE}', using gaussian")
            single_pulse = generate_gaussian_pulse(Config.IMPULSE_WIDTH, Config.IMPULSE_AMPLITUDE)

        
        pulse_length = len(single_pulse)
        impulse[:pulse_length] = single_pulse

        
        num_repetitions = Config.NUM_PULSES
        self.impulse_signal = np.tile(impulse, num_repetitions).astype(np.complex64)

        print(f"Generated {num_repetitions} impulses")
        print(f"Impulse width: {Config.IMPULSE_WIDTH} samples ({Config.IMPULSE_WIDTH/Config.TX_SAMPLE_RATE*1e9:.2f} ns)")
        print(f"Total signal length: {len(self.impulse_signal)} samples")

        return self.impulse_signal

    def capture_raw_iq_data(self, num_samples):
        """
        Capture raw IQ data from receiver

        Args:
            num_samples: Number of samples to capture

        Returns:
            Complex numpy array of IQ samples
        """
        
        rx_buff = np.empty(2 * num_samples, np.int16)

        
        sr = self.sdr.readStream(self.rx_stream, [rx_buff], num_samples,
                                timeoutUs=Config.RX_TIMEOUT_US)
        rc = sr.ret

        if rc != num_samples:
            print(f"Warning: Expected {num_samples} samples, got {rc}")

        
        s0 = rx_buff.astype(float) / np.power(2.0, Config.RX_BITS - 1)
        iq_data = (s0[::2] + 1j * s0[1::2])

        return iq_data

    def record_pulse_sequence(self):
        """
        Record a sequence of pulses with synchronized TX and RX
        Returns raw IQ data for each pulse
        """
        if self.sdr is None:
            raise RuntimeError("Hardware not initialized. Cannot record pulse sequence.")

        print(f"\nRecording {Config.NUM_PULSES} pulses...")

        
        self.sdr.activateStream(self.tx_stream)
        self.sdr.activateStream(self.rx_stream)
        self.streams_active = True

        
        samples_per_pulse = Config.SAMPLES_PER_PULSE
        total_samples = Config.NUM_PULSES * samples_per_pulse

        all_iq_data = []

        try:
            
            print("Starting transmission...")

            for pulse_idx in range(Config.NUM_PULSES):
                
                samples_per_interval = int(Config.TX_SAMPLE_RATE / Config.PULSE_REPETITION_FREQ)
                impulse = self.impulse_signal[pulse_idx * samples_per_interval:(pulse_idx + 1) * samples_per_interval]

                
                sr_tx = self.sdr.writeStream(self.tx_stream, [impulse], len(impulse))

                
                iq_data = self.capture_raw_iq_data(samples_per_pulse)
                all_iq_data.append(iq_data)

                
                if (pulse_idx + 1) % 100 == 0:
                    print(f"  Captured {pulse_idx + 1}/{Config.NUM_PULSES} pulses")

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            
            self.sdr.deactivateStream(self.tx_stream)
            self.sdr.deactivateStream(self.rx_stream)
            self.streams_active = False

        print(f"Capture complete: {len(all_iq_data)} pulses recorded")

        
        raw_data = np.array(all_iq_data)

        return raw_data

    def record_continuous(self, duration_seconds):
        """
        Record continuous data for offline pulse extraction

        Transmits impulse train continuously throughout the capture duration.

        Args:
            duration_seconds: Recording duration in seconds

        Returns:
            Raw IQ data array
        """
        if self.sdr is None:
            raise RuntimeError("Hardware not initialized. Cannot record continuous data.")
        if self.impulse_signal is None:
            raise RuntimeError("Impulse signal not generated. Call generate_impulse_signal() first.")

        print(f"\nRecording continuous data for {duration_seconds} seconds...")

        total_samples = int(Config.RX_SAMPLE_RATE * duration_seconds)

        # Build a single PRF-period containing one pulse at the start
        tx_samples_per_period = int(Config.TX_SAMPLE_RATE / Config.PULSE_REPETITION_FREQ)
        base_interval = np.zeros(tx_samples_per_period, dtype=np.complex64)

        # Use the existing generated pulse shape for one interval
        pulse_len = min(len(self.impulse_signal), tx_samples_per_period)
        base_interval[:pulse_len] = self.impulse_signal[:pulse_len]

        # Repeat this interval for the desired duration
        num_periods = int(duration_seconds * Config.PULSE_REPETITION_FREQ) + 1
        tx_signal = np.tile(base_interval, num_periods)
        print(f"  TX pulse train: {num_periods} pulses, {len(tx_signal)} samples total")

        
        self.sdr.activateStream(self.tx_stream)
        self.sdr.activateStream(self.rx_stream)
        self.streams_active = True

        all_data = []
        tx_offset = 0

        try:
            
            chunk_size = 16384
            tx_chunk_size = chunk_size  
            num_chunks = total_samples // chunk_size
            remaining = total_samples % chunk_size

            print("Starting continuous TX/RX...")

            for i in range(num_chunks):
                
                if tx_offset < len(tx_signal):
                    tx_end = min(tx_offset + tx_chunk_size, len(tx_signal))
                    tx_chunk = tx_signal[tx_offset:tx_end]
                    self.sdr.writeStream(self.tx_stream, [tx_chunk], len(tx_chunk))
                    tx_offset = tx_end

                
                iq_data = self.capture_raw_iq_data(chunk_size)
                all_data.append(iq_data)

                if (i + 1) % 100 == 0:
                    print(f"  Captured {(i+1)*chunk_size/Config.RX_SAMPLE_RATE:.1f}s / {duration_seconds}s")

            
            if remaining > 0:
                
                if tx_offset < len(tx_signal):
                    tx_end = min(tx_offset + remaining, len(tx_signal))
                    tx_chunk = tx_signal[tx_offset:tx_end]
                    self.sdr.writeStream(self.tx_stream, [tx_chunk], len(tx_chunk))

                iq_data = self.capture_raw_iq_data(remaining)
                all_data.append(iq_data)

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            self.sdr.deactivateStream(self.tx_stream)
            self.sdr.deactivateStream(self.rx_stream)
            self.streams_active = False

        print("Capture complete")

        
        raw_data = np.concatenate(all_data)

        return raw_data

    def cleanup(self):
        """Close streams and release SDR resources"""
        print("Cleaning up SDR resources...")

        if self.tx_stream is not None:
            self.sdr.closeStream(self.tx_stream)

        if self.rx_stream is not None:
            self.sdr.closeStream(self.rx_stream)

        self.streams_active = False

        print("Cleanup complete")

    def record_single_pulse(self):
        """
        Transmit one impulse interval and capture a single pulse worth of samples.

        Returns:
            1D complex array of length Config.SAMPLES_PER_PULSE
        """
        if self.sdr is None:
            raise RuntimeError("Hardware not initialized. Cannot record pulse.")
        if self.impulse_signal is None:
            raise RuntimeError("Impulse signal not generated. Call generate_impulse_signal() first.")
        if self.tx_stream is None or self.rx_stream is None:
            raise RuntimeError("Streams not set up. Call setup_transmitter()/setup_receiver() first.")

        # Activate streams once
        if not self.streams_active:
            self.sdr.activateStream(self.tx_stream)
            self.sdr.activateStream(self.rx_stream)
            self.streams_active = True

        samples_per_interval = int(Config.TX_SAMPLE_RATE / Config.PULSE_REPETITION_FREQ)
        interval = self.impulse_signal[:samples_per_interval]

        # Transmit one interval then capture one pulse worth of RX samples
        self.sdr.writeStream(self.tx_stream, [interval], len(interval))
        iq_data = self.capture_raw_iq_data(Config.SAMPLES_PER_PULSE)
        return iq_data


def test_sdr_capture():
    """Test function for SDR capture"""
    print("Testing SDR Capture Module")
    print("=" * 60)

    capture = SDRCapture()

    try:
        
        capture.setup_transmitter()
        capture.setup_receiver()

        
        capture.generate_impulse_signal()

        
        raw_data = capture.record_pulse_sequence()

        print(f"\nCaptured data shape: {raw_data.shape}")
        print(f"Data type: {raw_data.dtype}")

        
        if Config.SAVE_RAW_DATA:
            import os
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            np.save(f"{Config.OUTPUT_DIR}/raw_iq_data.npy", raw_data)
            print(f"Saved raw data to {Config.OUTPUT_DIR}/raw_iq_data.npy")

    finally:
        capture.cleanup()


if __name__ == "__main__":
    Config.print_config()
    test_sdr_capture()
