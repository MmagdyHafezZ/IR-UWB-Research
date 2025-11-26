"""
Respiration Extraction Module
Extracts breathing signal and estimates breathing rate using time-domain and frequency-domain methods
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from config import Config
from vmd import vmd, select_respiration_mode


class RespirationExtractor:
    """Extract and analyze respiration signals"""

    def __init__(self, chest_signal, sampling_rate=None):
        """
        Initialize respiration extractor

        Args:
            chest_signal: 1D complex array from chest range bin
            sampling_rate: Sampling rate in Hz (default: PRF from config)
        """
        self.chest_signal = chest_signal
        self.sampling_rate = sampling_rate if sampling_rate is not None else Config.PULSE_REPETITION_FREQ
        self.filtered_signal = None
        self.breathing_rate = None
        self.breathing_rate_fft = None

        
        self.vmd_modes = None
        self.vmd_omega = None
        self.vmd_respiration_mode = None
        self.vmd_mode_index = None
        self.vmd_mode_info = None
        self.vmd_breathing_rate = None
        self.vmd_breathing_rate_fft = None

    def apply_breathing_bandpass_filter(self, freq_min=None, freq_max=None, order=None):
        """
        Apply band-pass filter to isolate breathing signal

        Args:
            freq_min: Minimum breathing frequency in Hz (default from config)
            freq_max: Maximum breathing frequency in Hz (default from config)
            order: Filter order (default from config)

        Returns:
            Filtered breathing signal
        """
        if freq_min is None:
            freq_min = Config.BREATHING_FREQ_MIN
        if freq_max is None:
            freq_max = Config.BREATHING_FREQ_MAX
        if order is None:
            order = Config.FILTER_ORDER

        print(f"Applying breathing band-pass filter ({freq_min}-{freq_max} Hz, order={order})...")

        
        nyquist = self.sampling_rate / 2
        low = freq_min / nyquist
        high = freq_max / nyquist

        if high >= 1.0:
            print(f"Warning: Upper frequency {freq_max} Hz is above Nyquist frequency {nyquist} Hz")
            high = 0.99

        b, a = signal.butter(order, [low, high], btype='band', analog=False)

        
        phase_signal = np.unwrap(np.angle(self.chest_signal))

        
        self.filtered_signal = signal.filtfilt(b, a, phase_signal)

        print(f"Filtered signal shape: {self.filtered_signal.shape}")
        print(f"Filtered signal range: [{np.min(self.filtered_signal):.3f}, {np.max(self.filtered_signal):.3f}]")

        return self.filtered_signal

    def detect_breathing_rate_time_domain(self):
        """
        Detect breathing rate using peak detection in time domain

        Returns:
            Breathing rate in breaths per minute (BPM)
        """
        print("Detecting breathing rate using time-domain method...")

        if self.filtered_signal is None:
            self.apply_breathing_bandpass_filter()

        
        
        peak_prominence = np.std(self.filtered_signal) * 0.5
        peaks, properties = signal.find_peaks(self.filtered_signal,
                                             prominence=peak_prominence,
                                             distance=int(self.sampling_rate * 1.0))  

        num_peaks = len(peaks)
        print(f"Detected {num_peaks} breathing peaks")

        if num_peaks < 2:
            print("Warning: Not enough peaks detected for reliable breathing rate estimation")
            self.breathing_rate = 0
            return 0

        
        duration_seconds = len(self.filtered_signal) / self.sampling_rate

        
        self.breathing_rate = (num_peaks / duration_seconds) * 60

        print(f"Breathing rate (time-domain): {self.breathing_rate:.1f} BPM")

        return self.breathing_rate

    def detect_breathing_rate_frequency_domain(self):
        """
        Detect breathing rate using FFT in frequency domain

        Returns:
            Breathing rate in breaths per minute (BPM)
        """
        print("Detecting breathing rate using frequency-domain method...")

        if self.filtered_signal is None:
            self.apply_breathing_bandpass_filter()

        
        N = len(self.filtered_signal)
        yf = fft(self.filtered_signal)
        xf = fftfreq(N, 1 / self.sampling_rate)

        
        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])

        
        breathing_range_idx = (xf_positive >= Config.BREATHING_FREQ_MIN) & (xf_positive <= Config.BREATHING_FREQ_MAX)
        xf_breathing = xf_positive[breathing_range_idx]
        yf_breathing = yf_positive[breathing_range_idx]

        if len(yf_breathing) == 0:
            print("Warning: No frequencies found in breathing range")
            self.breathing_rate_fft = 0
            return 0

        
        dominant_idx = np.argmax(yf_breathing)
        dominant_freq = xf_breathing[dominant_idx]
        dominant_magnitude = yf_breathing[dominant_idx]

        
        mean_magnitude = np.mean(yf_breathing)
        magnitude_ratio = dominant_magnitude / (mean_magnitude + 1e-10)

        
        if magnitude_ratio < 2.0:
            print("Warning: No significant breathing peak detected in spectrum")
            self.breathing_rate_fft = 0
            return 0

        
        self.breathing_rate_fft = dominant_freq * 60

        print(f"Breathing rate (frequency-domain): {self.breathing_rate_fft:.1f} BPM")
        print(f"Dominant frequency: {dominant_freq:.3f} Hz")

        return self.breathing_rate_fft

    def get_breathing_waveform(self):
        """
        Get the filtered breathing waveform

        Returns:
            Tuple of (time_axis, waveform)
        """
        if self.filtered_signal is None:
            self.apply_breathing_bandpass_filter()

        time_axis = np.arange(len(self.filtered_signal)) / self.sampling_rate

        return time_axis, self.filtered_signal

    def get_frequency_spectrum(self):
        """
        Get the frequency spectrum of the breathing signal

        Returns:
            Tuple of (frequencies, magnitude)
        """
        if self.filtered_signal is None:
            self.apply_breathing_bandpass_filter()

        
        N = len(self.filtered_signal)
        yf = fft(self.filtered_signal)
        xf = fftfreq(N, 1 / self.sampling_rate)

        
        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])

        return xf_positive, yf_positive

    def estimate_breathing_quality(self):
        """
        Estimate the quality of breathing signal detection

        Returns:
            Quality score between 0 (poor) and 1 (excellent)
        """
        print("Estimating breathing signal quality...")

        if self.filtered_signal is None:
            self.apply_breathing_bandpass_filter()

        
        N = len(self.filtered_signal)
        yf = fft(self.filtered_signal)
        xf = fftfreq(N, 1 / self.sampling_rate)

        
        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])

        
        breathing_idx = (xf_positive >= Config.BREATHING_FREQ_MIN) & (xf_positive <= Config.BREATHING_FREQ_MAX)
        in_band_power = np.sum(yf_positive[breathing_idx]**2)

        
        out_band_idx = ~breathing_idx
        out_band_power = np.sum(yf_positive[out_band_idx]**2)

        if out_band_power > 0 and in_band_power > 0:
            snr = 10 * np.log10(in_band_power / out_band_power)
        elif in_band_power > 0:
            snr = 100  
        else:
            snr = 0  

        
        quality = np.clip(snr / 20.0, 0, 1)

        print(f"Signal quality: {quality:.2f} (SNR: {snr:.1f} dB)")

        return quality

    def analyze_breathing_pattern(self):
        """
        Analyze breathing pattern for regularity and depth

        Returns:
            Dictionary with breathing pattern metrics
        """
        print("Analyzing breathing pattern...")

        if self.filtered_signal is None:
            self.apply_breathing_bandpass_filter()

        
        peaks, _ = signal.find_peaks(self.filtered_signal,
                                    prominence=np.std(self.filtered_signal) * 0.3)
        troughs, _ = signal.find_peaks(-self.filtered_signal,
                                       prominence=np.std(self.filtered_signal) * 0.3)

        
        if len(peaks) > 0 and len(troughs) > 0:
            peak_values = self.filtered_signal[peaks]
            trough_values = self.filtered_signal[troughs]
            breathing_depth = np.mean(peak_values) - np.mean(trough_values)
        else:
            breathing_depth = 0

        
        if len(peaks) > 1:
            inter_breath_intervals = np.diff(peaks) / self.sampling_rate
            regularity = 1.0 - (np.std(inter_breath_intervals) / np.mean(inter_breath_intervals))
            regularity = np.clip(regularity, 0, 1)
        else:
            regularity = 0

        
        if len(peaks) > 1 and len(troughs) > 1:
            
            all_events = []
            for p in peaks:
                all_events.append((p, 'peak'))
            for t in troughs:
                all_events.append((t, 'trough'))
            all_events.sort(key=lambda x: x[0])

            
            inhale_times = []
            exhale_times = []

            for i in range(len(all_events) - 1):
                curr_idx, curr_type = all_events[i]
                next_idx, next_type = all_events[i + 1]
                duration = (next_idx - curr_idx) / self.sampling_rate

                if curr_type == 'trough' and next_type == 'peak':
                    
                    inhale_times.append(duration)
                elif curr_type == 'peak' and next_type == 'trough':
                    
                    exhale_times.append(duration)

            if len(inhale_times) > 0 and len(exhale_times) > 0:
                avg_inhale = np.mean(inhale_times)
                avg_exhale = np.mean(exhale_times)
                ie_ratio = avg_inhale / (avg_exhale + 1e-6)
            else:
                ie_ratio = 1.0
        else:
            ie_ratio = 1.0

        metrics = {
            'breathing_depth': breathing_depth,
            'regularity': regularity,
            'ie_ratio': ie_ratio,
            'num_breaths': len(peaks)
        }

        print(f"Breathing depth: {breathing_depth:.3f}")
        print(f"Regularity: {regularity:.2f}")
        print(f"I/E ratio: {ie_ratio:.2f}")
        print(f"Number of breaths: {len(peaks)}")

        return metrics

    def apply_vmd_decomposition(self, alpha=None, K=None, tau=None, DC=None, init=None, tol=None, max_iter=None):
        """
        Apply Variational Mode Decomposition to extract breathing signal

        VMD decomposes the signal into K narrowband modes using ADMM optimization.
        The mode with dominant frequency in breathing range is selected as respiration.

        Args:
            alpha: Bandwidth penalty (default from config)
            K: Number of modes (default from config)
            tau: Noise tolerance (default from config)
            DC: Include DC component (default from config)
            init: Initialization method (default from config)
            tol: Convergence tolerance (default from config)
            max_iter: Maximum iterations (default from config)

        Returns:
            Respiration signal extracted from VMD mode
        """
        
        if alpha is None:
            alpha = Config.VMD_ALPHA
        if K is None:
            K = Config.VMD_NUM_MODES
        if tau is None:
            tau = Config.VMD_TAU
        if DC is None:
            DC = Config.VMD_DC_PART
        if init is None:
            init = Config.VMD_INIT_METHOD
        if tol is None:
            tol = Config.VMD_TOL
        if max_iter is None:
            max_iter = Config.VMD_MAX_ITER

        print(f"Applying VMD decomposition (K={K}, alpha={alpha}, tol={tol})...")

        
        phase_signal = np.unwrap(np.angle(self.chest_signal))

        
        self.vmd_modes, vmd_modes_fft, self.vmd_omega = vmd(
            signal=phase_signal,
            alpha=alpha,
            tau=tau,
            K=K,
            DC=DC,
            init=init,
            tol=tol,
            max_iter=max_iter
        )

        print(f"VMD decomposition complete: {K} modes extracted")
        print(f"Mode center frequencies: {[f'{omega*self.sampling_rate:.3f} Hz' for omega in self.vmd_omega]}")

        
        self.vmd_respiration_mode, self.vmd_mode_index, self.vmd_mode_info = select_respiration_mode(
            modes=self.vmd_modes,
            omega=self.vmd_omega,
            fs=self.sampling_rate,
            breathing_freq_min=Config.BREATHING_FREQ_MIN,
            breathing_freq_max=Config.BREATHING_FREQ_MAX
        )

        print(f"Selected mode {self.vmd_mode_index} as respiration component")
        print(f"  Center frequency: {self.vmd_omega[self.vmd_mode_index]*self.sampling_rate:.3f} Hz")
        print(f"  Breathing power ratio: {self.vmd_mode_info[self.vmd_mode_index]['breathing_power_ratio']:.3f}")

        return self.vmd_respiration_mode

    def detect_breathing_rate_vmd_time_domain(self):
        """
        Detect breathing rate from VMD mode using time-domain peak detection

        Returns:
            Breathing rate in breaths per minute (BPM)
        """
        print("Detecting breathing rate from VMD (time-domain)...")

        if self.vmd_respiration_mode is None:
            self.apply_vmd_decomposition()

        
        peak_prominence = np.std(self.vmd_respiration_mode) * 0.5
        peaks, properties = signal.find_peaks(
            self.vmd_respiration_mode,
            prominence=peak_prominence,
            distance=int(self.sampling_rate * 1.0)  
        )

        num_peaks = len(peaks)
        print(f"Detected {num_peaks} breathing peaks in VMD mode")

        if num_peaks < 2:
            print("Warning: Not enough peaks detected in VMD mode")
            self.vmd_breathing_rate = 0
            return 0

        
        duration_seconds = len(self.vmd_respiration_mode) / self.sampling_rate
        self.vmd_breathing_rate = (num_peaks / duration_seconds) * 60

        print(f"Breathing rate (VMD time-domain): {self.vmd_breathing_rate:.1f} BPM")

        return self.vmd_breathing_rate

    def detect_breathing_rate_vmd_frequency_domain(self):
        """
        Detect breathing rate from VMD mode using FFT

        Returns:
            Breathing rate in breaths per minute (BPM)
        """
        print("Detecting breathing rate from VMD (frequency-domain)...")

        if self.vmd_respiration_mode is None:
            self.apply_vmd_decomposition()

        
        N = len(self.vmd_respiration_mode)
        yf = fft(self.vmd_respiration_mode)
        xf = fftfreq(N, 1 / self.sampling_rate)

        
        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])

        
        breathing_range_idx = (xf_positive >= Config.BREATHING_FREQ_MIN) & (xf_positive <= Config.BREATHING_FREQ_MAX)
        xf_breathing = xf_positive[breathing_range_idx]
        yf_breathing = yf_positive[breathing_range_idx]

        if len(yf_breathing) == 0:
            print("Warning: No frequencies in breathing range for VMD mode")
            self.vmd_breathing_rate_fft = 0
            return 0

        
        dominant_idx = np.argmax(yf_breathing)
        dominant_freq = xf_breathing[dominant_idx]
        dominant_magnitude = yf_breathing[dominant_idx]

        
        mean_magnitude = np.mean(yf_breathing)
        magnitude_ratio = dominant_magnitude / (mean_magnitude + 1e-10)

        if magnitude_ratio < 2.0:
            print("Warning: No significant breathing peak in VMD mode spectrum")
            self.vmd_breathing_rate_fft = 0
            return 0

        
        self.vmd_breathing_rate_fft = dominant_freq * 60

        print(f"Breathing rate (VMD frequency-domain): {self.vmd_breathing_rate_fft:.1f} BPM")
        print(f"Dominant frequency: {dominant_freq:.3f} Hz")

        return self.vmd_breathing_rate_fft

    def get_vmd_mode_comparison(self):
        """
        Get comparison information for all VMD modes

        Returns:
            List of dictionaries with mode information
        """
        if self.vmd_mode_info is None:
            print("Warning: VMD not yet applied")
            return []

        return self.vmd_mode_info

    def run_full_analysis(self):
        """
        Run complete respiration analysis

        Performs both baseline (bandpass filter) and VMD-based analysis if enabled.

        Returns:
            Dictionary with all analysis results including comparison
        """
        print("\n" + "=" * 60)
        print("Running Full Respiration Analysis")
        print("=" * 60)

        
        
        
        print("\n--- Baseline Method (Bandpass Filter) ---")

        
        self.apply_breathing_bandpass_filter()

        
        rate_time = self.detect_breathing_rate_time_domain()
        rate_freq = self.detect_breathing_rate_frequency_domain()

        
        if rate_time > 0 and rate_freq > 0:
            rate_avg = (rate_time + rate_freq) / 2
        elif rate_time > 0:
            rate_avg = rate_time
        elif rate_freq > 0:
            rate_avg = rate_freq
        else:
            rate_avg = 0

        
        quality = self.estimate_breathing_quality()

        
        pattern_metrics = self.analyze_breathing_pattern()

        
        
        
        vmd_results = None
        if Config.USE_VMD:
            print("\n--- VMD Method (Variational Mode Decomposition) ---")

            try:
                
                self.apply_vmd_decomposition()

                
                vmd_rate_time = self.detect_breathing_rate_vmd_time_domain()
                vmd_rate_freq = self.detect_breathing_rate_vmd_frequency_domain()

                
                if vmd_rate_time > 0 and vmd_rate_freq > 0:
                    vmd_rate_avg = (vmd_rate_time + vmd_rate_freq) / 2
                elif vmd_rate_time > 0:
                    vmd_rate_avg = vmd_rate_time
                elif vmd_rate_freq > 0:
                    vmd_rate_avg = vmd_rate_freq
                else:
                    vmd_rate_avg = 0

                vmd_results = {
                    'breathing_rate_time': vmd_rate_time,
                    'breathing_rate_freq': vmd_rate_freq,
                    'breathing_rate_avg': vmd_rate_avg,
                    'selected_mode_index': self.vmd_mode_index,
                    'mode_center_freq': self.vmd_omega[self.vmd_mode_index] * self.sampling_rate,
                    'mode_info': self.vmd_mode_info
                }

            except Exception as e:
                print(f"Warning: VMD analysis failed: {e}")
                vmd_results = None

        
        
        
        results = {
            'baseline': {
                'breathing_rate_time': rate_time,
                'breathing_rate_freq': rate_freq,
                'breathing_rate_avg': rate_avg,
                'signal_quality': quality,
                'pattern_metrics': pattern_metrics
            },
            'vmd': vmd_results
        }

        
        
        
        print("\n" + "=" * 60)
        print("Respiration Analysis Complete")
        print("=" * 60)
        print("Baseline Method:")
        print(f"  Breathing Rate: {rate_avg:.1f} BPM")
        print(f"  Signal Quality: {quality:.2f}")

        if vmd_results is not None:
            print("\nVMD Method:")
            print(f"  Breathing Rate: {vmd_results['breathing_rate_avg']:.1f} BPM")
            print(f"  Selected Mode: {vmd_results['selected_mode_index']} "
                  f"(center freq: {vmd_results['mode_center_freq']:.3f} Hz)")

            
            if rate_avg > 0 and vmd_results['breathing_rate_avg'] > 0:
                diff = abs(rate_avg - vmd_results['breathing_rate_avg'])
                print(f"\nComparison:")
                print(f"  Difference: {diff:.1f} BPM")
                print(f"  Agreement: {'Good' if diff < 2.0 else 'Poor'}")

        print("=" * 60)

        return results


def test_respiration_extractor():
    """Test function for respiration extractor"""
    print("Testing Respiration Extractor Module")
    print("=" * 60)

    
    print("Generating synthetic breathing signal...")

    duration = 30  
    fs = Config.PULSE_REPETITION_FREQ
    t = np.arange(0, duration, 1/fs)

    
    breathing_rate_bpm = 15  
    breathing_freq = breathing_rate_bpm / 60  

    
    breathing_amplitude = 0.5
    breathing_signal = breathing_amplitude * np.sin(2 * np.pi * breathing_freq * t)

    
    breathing_signal += 0.1 * np.sin(2 * np.pi * 2 * breathing_freq * t)

    
    noise = 0.05 * np.random.randn(len(t))
    breathing_signal += noise

    
    phase = breathing_signal
    chest_signal = np.exp(1j * phase)

    print(f"Synthetic signal length: {len(chest_signal)} samples ({duration} seconds)")
    print(f"Expected breathing rate: {breathing_rate_bpm} BPM")

    
    extractor = RespirationExtractor(chest_signal, sampling_rate=fs)
    results = extractor.run_full_analysis()

    print(f"\nDetected breathing rate: {results['breathing_rate_avg']:.1f} BPM")
    print(f"Error: {abs(results['breathing_rate_avg'] - breathing_rate_bpm):.1f} BPM")

    
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    time_axis, waveform = extractor.get_breathing_waveform()
    np.save(f"{Config.OUTPUT_DIR}/breathing_waveform.npy", waveform)
    np.save(f"{Config.OUTPUT_DIR}/breathing_time_axis.npy", time_axis)

    frequencies, spectrum = extractor.get_frequency_spectrum()
    np.save(f"{Config.OUTPUT_DIR}/frequency_spectrum.npy", spectrum)
    np.save(f"{Config.OUTPUT_DIR}/frequencies.npy", frequencies)

    
    import json
    with open(f"{Config.OUTPUT_DIR}/respiration_results.json", 'w') as f:
        
        json_results = {
            'breathing_rate_time': float(results['breathing_rate_time']),
            'breathing_rate_freq': float(results['breathing_rate_freq']),
            'breathing_rate_avg': float(results['breathing_rate_avg']),
            'signal_quality': float(results['signal_quality']),
            'pattern_metrics': {k: float(v) for k, v in results['pattern_metrics'].items()}
        }
        json.dump(json_results, f, indent=4)

    print(f"\nSaved results to {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    Config.print_config()
    test_respiration_extractor()
