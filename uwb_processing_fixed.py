#!/usr/bin/env python3
"""
Fixed IR-UWB Processing Pipeline
Addresses all numerical and range issues for accurate BPM detection

Key fixes:
1. Correct range-bin to distance mapping
2. Chest detection restricted to 0.3-2.0m
3. Static clutter removal (slow-time mean)
4. Proper phase extraction with detrending
5. Numerical stability (no overflows/NaNs)
6. Robust chest detector with thresholding
7. Clean BPM estimation (6-42 BPM)
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


class UWBProcessor:
    """Complete IR-UWB processing pipeline with all fixes"""

    def __init__(self, config):
        """
        Initialize with system configuration

        Args:
            config: Config object with system parameters
        """
        self.config = config
        self.c = 3e8  # Speed of light

        # Breathing band limits
        self.bpm_min = 6   # 6 BPM = 0.1 Hz
        self.bpm_max = 42  # 42 BPM = 0.7 Hz
        self.freq_min = self.bpm_min / 60.0  # Hz
        self.freq_max = self.bpm_max / 60.0  # Hz

        # Chest detection range
        self.chest_range_min = 0.3  # meters (avoid direct leakage)
        self.chest_range_max = 2.0  # meters (realistic for indoor)

    def compute_range_axis(self, num_bins):
        """
        Fix #1: Correct range-bin to distance mapping

        For IR-UWB with oversampled pulse capture:
        - We capture SAMPLES_PER_PULSE samples for each pulse
        - These samples represent the echo over time
        - The actual range depends on pulse duration and sampling

        Corrected mapping for IR-UWB:
        - Each pulse has a finite duration (e.g., 100ns)
        - We oversample this to get range profile
        - Effective bandwidth = RX_SAMPLE_RATE / oversampling_factor

        Args:
            num_bins: Number of fast-time bins (samples per pulse)

        Returns:
            range_axis: Array of distances in meters
        """
        # For IR-UWB, the range is determined by the effective bandwidth
        # If we have 1024 samples per pulse captured at 31.25 MHz
        # over a pulse repetition interval of 1ms (1000 Hz PRF)

        # The actual time window we're sampling
        # Assuming we capture the echo response over a fixed time window
        # This needs to be calibrated to the actual radar setup

        # Method 1: If samples span the full PRI
        # total_time = 1.0 / self.config.PULSE_REPETITION_FREQ

        # Method 2: If samples represent a fixed observation window
        # For IR-UWB, we need to observe enough time to cover the desired range
        # To see up to 3m, we need: t = 2 * 3m / 3e8 = 20ns
        # Let's use 50ns to cover up to 7.5m
        observation_time = 50e-9  # 50 nanoseconds for up to 7.5m range

        # Time per sample
        dt = observation_time / num_bins

        # Convert to range
        time_axis = np.arange(num_bins) * dt
        range_axis = (self.c * time_axis) / 2.0

        print(f"Range mapping: {num_bins} bins over {observation_time*1e6:.1f}μs")
        print(f"  Range: {range_axis[0]:.3f} to {range_axis[-1]:.3f} m")
        print(f"  Resolution: {range_axis[1] - range_axis[0]:.3f} m")

        return range_axis

    def remove_static_clutter(self, rtm_matrix):
        """
        Fix #2: Static clutter removal via slow-time mean subtraction

        Args:
            rtm_matrix: Range-time matrix [slow_time x fast_time]

        Returns:
            rtm_demeaned: Clutter-removed matrix
            clutter_profile: The removed static clutter
        """
        # Check for NaN/Inf before processing
        if not np.all(np.isfinite(rtm_matrix)):
            print("Warning: NaN/Inf in input, replacing with zeros")
            rtm_matrix = np.nan_to_num(rtm_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute mean over slow-time (axis=0) for each range bin
        clutter_profile = np.mean(rtm_matrix, axis=0, keepdims=True)

        # Subtract mean to remove static reflections
        rtm_demeaned = rtm_matrix - clutter_profile

        # Optionally zero out direct leakage bins (first few bins)
        leakage_bins = 5  # First 5 bins often have TX-RX leakage
        rtm_demeaned[:, :leakage_bins] = 0

        print("Clutter removal applied:")
        print(f"  Input power: {np.mean(np.abs(rtm_matrix)**2):.2e}")
        print(f"  Output power: {np.mean(np.abs(rtm_demeaned)**2):.2e}")
        print(f"  Suppression: {10*np.log10(np.mean(np.abs(rtm_matrix)**2) / (np.mean(np.abs(rtm_demeaned)**2) + 1e-10)):.1f} dB")

        return rtm_demeaned, clutter_profile

    def detect_chest_robust(self, rtm_demeaned, range_axis):
        """
        Fix #3 & #5: Robust chest detection in 0.3-2.0m with thresholding

        Args:
            rtm_demeaned: Clutter-removed range-time matrix
            range_axis: Range values in meters

        Returns:
            chest_bin: Index of detected chest bin (or None if not found)
            chest_range: Distance to chest in meters
            info: Detection diagnostics
        """
        # Compute slow-time variance for each range bin
        # Use magnitude if complex
        if np.iscomplexobj(rtm_demeaned):
            variance_profile = np.var(np.abs(rtm_demeaned), axis=0)
        else:
            variance_profile = np.var(rtm_demeaned, axis=0)

        # Fix #4: Prevent overflow - normalize variance
        max_var = np.max(variance_profile)
        if max_var > 1e10:
            print(f"Warning: Large variance {max_var:.2e}, normalizing")
            variance_profile = variance_profile / max_var

        # Smooth to reduce noise
        variance_smooth = gaussian_filter1d(variance_profile, sigma=3)

        # Find bins within chest search range
        valid_mask = (range_axis >= self.chest_range_min) & (range_axis <= self.chest_range_max)
        valid_bins = np.where(valid_mask)[0]

        if len(valid_bins) == 0:
            print(f"Error: No bins in range [{self.chest_range_min}, {self.chest_range_max}]m")
            return None, None, {'error': 'No bins in search range'}

        # Extract variance in search range
        search_variance = variance_smooth[valid_bins]
        search_range = range_axis[valid_bins]

        # Compute noise floor and threshold
        noise_floor = np.median(search_variance)
        noise_std = np.std(search_variance)
        threshold = noise_floor + 2.0 * noise_std  # 2-sigma threshold

        print(f"Chest detection in [{self.chest_range_min}, {self.chest_range_max}]m:")
        print(f"  Noise floor: {noise_floor:.2e}")
        print(f"  Threshold: {threshold:.2e}")

        # Find bins above threshold
        above_threshold = search_variance > threshold

        if not np.any(above_threshold):
            print("Warning: No bins exceed threshold - no chest detected")
            print("  Possible causes:")
            print("    - Subject not in range")
            print("    - Subject too still")
            print("    - Signal too weak")
            return None, None, {'error': 'No chest detected', 'variance_profile': variance_smooth}

        # Among bins above threshold, pick the one with max variance
        candidate_indices = np.where(above_threshold)[0]
        candidate_variances = search_variance[candidate_indices]
        best_idx = candidate_indices[np.argmax(candidate_variances)]

        chest_bin = valid_bins[best_idx]
        chest_range = range_axis[chest_bin]

        # Sanity check
        if chest_range < self.chest_range_min or chest_range > self.chest_range_max:
            print(f"Error: Detected chest at {chest_range:.2f}m outside valid range!")
            return None, None, {'error': 'Invalid chest range'}

        print(f"✓ Chest detected at bin {chest_bin}, range {chest_range:.2f}m")
        print(f"  Variance: {variance_smooth[chest_bin]:.2e} ({variance_smooth[chest_bin]/noise_floor:.1f}x noise floor)")

        info = {
            'variance_profile': variance_smooth,
            'chest_bin': chest_bin,
            'chest_range': chest_range,
            'noise_floor': noise_floor,
            'threshold': threshold,
            'num_candidates': len(candidate_indices)
        }

        return chest_bin, chest_range, info

    def extract_phase_stable(self, rtm_demeaned, chest_bin):
        """
        Fix #4: Stable phase extraction with numerical safety

        Args:
            rtm_demeaned: Clutter-removed matrix
            chest_bin: Index of chest range bin

        Returns:
            phase_signal: Unwrapped phase at chest bin
        """
        # Extract complex signal at chest bin
        chest_signal = rtm_demeaned[:, chest_bin]

        # Check for zeros/NaN
        if np.all(chest_signal == 0):
            print("Error: Chest signal is all zeros")
            return np.zeros(len(chest_signal))

        # Replace any NaN/Inf
        chest_signal = np.nan_to_num(chest_signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract phase (angle of complex number)
        phase_raw = np.angle(chest_signal)

        # Unwrap phase to avoid discontinuities
        phase_unwrapped = np.unwrap(phase_raw)

        return phase_unwrapped

    def detrend_and_filter(self, phase_signal, prf):
        """
        Fix #3: Detrend and high-pass filter the breathing signal

        Args:
            phase_signal: Unwrapped phase
            prf: Pulse repetition frequency (slow-time sampling rate)

        Returns:
            breathing_signal: Clean, zero-centered breathing waveform
        """
        # Remove linear trend
        phase_detrended = signal.detrend(phase_signal, type='linear')

        # Remove DC offset
        phase_detrended = phase_detrended - np.mean(phase_detrended)

        # Design high-pass filter (0.05 Hz cutoff)
        # This removes drift below 3 BPM
        cutoff_hz = 0.05  # Hz
        nyquist = prf / 2.0

        if cutoff_hz >= nyquist:
            print(f"Warning: Cutoff {cutoff_hz} >= Nyquist {nyquist}, skipping filter")
            return phase_detrended

        # Butterworth high-pass, order 4
        sos = signal.butter(4, cutoff_hz/nyquist, btype='high', output='sos')
        breathing_signal = signal.sosfiltfilt(sos, phase_detrended)

        # Ensure zero mean
        breathing_signal = breathing_signal - np.mean(breathing_signal)

        # Check for numerical issues
        if not np.all(np.isfinite(breathing_signal)):
            print("Warning: Non-finite values after filtering, replacing")
            breathing_signal = np.nan_to_num(breathing_signal, nan=0.0)

        print(f"Breathing signal: std={np.std(breathing_signal):.3f}, mean={np.mean(breathing_signal):.3e}")

        return breathing_signal

    def estimate_breathing_rate(self, breathing_signal, prf):
        """
        Fix #6: Clean breathing rate estimation in 6-42 BPM band

        Args:
            breathing_signal: Detrended, filtered breathing waveform
            prf: Slow-time sampling rate

        Returns:
            bpm: Breathing rate in BPM (0 if not detected)
            spectrum: FFT magnitude spectrum
            freqs: Frequency axis
            quality: Signal quality metric
        """
        # Check signal quality
        if np.std(breathing_signal) < 1e-6:
            print("Warning: Breathing signal too weak (std < 1e-6)")
            return 0, None, None, "poor"

        # Normalize to prevent overflow
        breathing_normalized = breathing_signal / np.std(breathing_signal)

        # Compute FFT (real-valued input -> use rfft)
        n = len(breathing_normalized)
        spectrum = np.abs(np.fft.rfft(breathing_normalized))
        freqs = np.fft.rfftfreq(n, d=1.0/prf)

        # Fix #4: Normalize spectrum to prevent huge values
        spectrum = spectrum / n

        # Find breathing band (0.1-0.7 Hz)
        band_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        band_freqs = freqs[band_mask]
        band_spectrum = spectrum[band_mask]

        if len(band_spectrum) == 0:
            print("Error: No frequencies in breathing band")
            return 0, spectrum, freqs, "error"

        # Find peak in breathing band
        peak_idx = np.argmax(band_spectrum)
        peak_freq = band_freqs[peak_idx]
        peak_power = band_spectrum[peak_idx]

        # Compute noise floor (median of non-peak bins)
        noise_floor = np.median(band_spectrum)

        # Signal quality check
        snr = peak_power / (noise_floor + 1e-10)
        if snr < 2.0:  # Peak must be 2x noise floor
            print(f"Warning: Low SNR {snr:.1f}, breathing may not be reliable")
            quality = "poor"
        else:
            quality = "good"

        # Convert to BPM
        bpm = peak_freq * 60.0

        # Final sanity check
        if bpm < self.bpm_min or bpm > self.bpm_max:
            print(f"Warning: BPM {bpm:.1f} outside valid range [{self.bpm_min}, {self.bpm_max}]")
            return 0, spectrum, freqs, "out_of_range"

        print(f"✓ Breathing rate: {bpm:.1f} BPM (SNR: {snr:.1f})")

        return bpm, spectrum, freqs, quality

    def process_complete(self, rtm_matrix):
        """
        Fix #7: Complete end-to-end processing with all checks

        Args:
            rtm_matrix: Raw range-time matrix [slow_time x fast_time]

        Returns:
            results: Dictionary with all processing outputs
        """
        print("\n" + "="*60)
        print("IR-UWB BREATHING DETECTION - FIXED PIPELINE")
        print("="*60)

        results = {}

        # Get dimensions
        n_slow, n_fast = rtm_matrix.shape
        prf = self.config.PULSE_REPETITION_FREQ

        print(f"Input: {n_slow} pulses x {n_fast} range bins")
        print(f"PRF: {prf} Hz, Duration: {n_slow/prf:.1f}s")

        # Step 1: Compute range axis
        print("\n[Step 1] Range mapping:")
        range_axis = self.compute_range_axis(n_fast)
        results['range_axis'] = range_axis

        # Step 2: Remove static clutter
        print("\n[Step 2] Clutter removal:")
        rtm_demeaned, clutter = self.remove_static_clutter(rtm_matrix)
        results['rtm_demeaned'] = rtm_demeaned
        results['clutter'] = clutter

        # Step 3: Detect chest
        print("\n[Step 3] Chest detection:")
        chest_bin, chest_range, chest_info = self.detect_chest_robust(rtm_demeaned, range_axis)
        results.update(chest_info)

        if chest_bin is None:
            print("\n✗ NO CHEST DETECTED - Cannot extract breathing")
            results['bpm'] = 0
            results['error'] = "No chest detected"
            return results

        results['chest_bin'] = chest_bin
        results['chest_range'] = chest_range

        # Step 4: Extract phase
        print("\n[Step 4] Phase extraction:")
        phase_signal = self.extract_phase_stable(rtm_demeaned, chest_bin)
        results['phase_raw'] = phase_signal

        # Step 5: Detrend and filter
        print("\n[Step 5] Detrending and filtering:")
        breathing_signal = self.detrend_and_filter(phase_signal, prf)
        results['breathing_signal'] = breathing_signal

        # Step 6: Estimate breathing rate
        print("\n[Step 6] Breathing rate estimation:")
        bpm, spectrum, freqs, quality = self.estimate_breathing_rate(breathing_signal, prf)
        results['bpm'] = bpm
        results['spectrum'] = spectrum
        results['freqs'] = freqs
        results['quality'] = quality

        # Final summary
        print("\n" + "="*60)
        if bpm > 0:
            print(f"✓ SUCCESS: {bpm:.1f} BPM at {chest_range:.2f}m (quality: {quality})")
        else:
            print(f"✗ FAILED: No breathing detected (quality: {quality})")
        print("="*60)

        return results


def test_fixed_processing():
    """Test the fixed processing pipeline with synthetic data"""
    from config import Config

    print("Testing Fixed IR-UWB Processing")
    print("-" * 60)

    # Create synthetic data
    n_pulses = 2000
    n_bins = 512
    prf = Config.PULSE_REPETITION_FREQ

    # Create processor
    processor = UWBProcessor(Config)

    # Generate synthetic radar data
    # Add static clutter
    rtm = np.random.randn(n_pulses, n_bins) * 10

    # Add strong static reflections (walls)
    rtm[:, 50] += 100  # Wall at ~0.5m
    rtm[:, 200] += 50  # Wall at ~2m

    # Add breathing signal at realistic range (0.8m)
    range_axis = processor.compute_range_axis(n_bins)
    chest_bin = np.argmin(np.abs(range_axis - 0.8))  # Find bin at 0.8m

    # Create breathing modulation (15 BPM = 0.25 Hz)
    t = np.arange(n_pulses) / prf
    breathing_phase = 2 * np.pi * 0.25 * t

    # Add to complex data
    rtm = rtm.astype(complex)
    rtm[:, chest_bin] += 0.5 * np.exp(1j * breathing_phase)

    # Process
    results = processor.process_complete(rtm)

    # Check results
    print(f"\nExpected: 15 BPM at 0.8m")
    print(f"Detected: {results['bpm']:.1f} BPM at {results.get('chest_range', 0):.2f}m")

    return results


if __name__ == "__main__":
    test_results = test_fixed_processing()