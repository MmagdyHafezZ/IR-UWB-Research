"""
Processing Fixes for IR-UWB Respiration Detection
Addresses issues identified from diagnostic plots
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from config import Config


def improved_chest_detection(variance_profile, range_bins, smoothing_sigma=5.0,
                             prominence_factor=2.0, min_range_m=0.3, max_range_m=3.0):
    """
    Improved chest detection with smoothing and peak prominence

    Fixes issues:
    - Noisy variance profile with false peaks
    - Suspicious sharp spikes
    - No clean peak structure

    Args:
        variance_profile: 1D array of variance values per range bin
        range_bins: 1D array of range values in meters
        smoothing_sigma: Gaussian smoothing sigma (higher = smoother)
        prominence_factor: Peak must be this many times the median
        min_range_m: Minimum plausible chest range (meters)
        max_range_m: Maximum plausible chest range (meters)

    Returns:
        chest_bin: Index of detected chest bin
        info: Dictionary with diagnostic information
    """

    # Ensure variance_profile is real-valued (gaussian_filter1d doesn't support complex)
    if np.iscomplexobj(variance_profile):
        print("Warning: variance_profile is complex, taking magnitude")
        variance_profile = np.abs(variance_profile)

    # Step 1: Smooth variance profile to reduce noise
    smoothed_variance = gaussian_filter1d(variance_profile, sigma=smoothing_sigma)

    # Step 2: Define search range (plausible chest locations)
    valid_idx = (range_bins >= min_range_m) & (range_bins <= max_range_m)
    search_bins = np.where(valid_idx)[0]

    if len(search_bins) == 0:
        print("Warning: No bins in valid range, using full profile")
        search_bins = np.arange(len(variance_profile))

    # Step 3: Find peaks with prominence requirement
    search_variance = smoothed_variance[search_bins]
    median_var = np.median(search_variance)
    std_var = np.std(search_variance)

    # Check for flat variance profile (no variation)
    if std_var < 1e-10:
        print("Warning: Variance profile is flat (std < 1e-10)")
        print("  No chest reflection detected - this usually means:")
        print("    - Subject too far from radar")
        print("    - No subject present")
        print("    - Signal quality too low")
        # Use middle of search range as fallback
        fallback_idx = len(search_bins) // 2
        chest_bin = search_bins[fallback_idx]
        info = {
            'chest_range': range_bins[chest_bin],
            'smoothed_variance': smoothed_variance,
            'prominence_threshold': 0,
            'num_candidates': 0,
            'warning': 'Flat variance - no chest detected'
        }
        print(f"  Using fallback: bin {chest_bin} ({range_bins[chest_bin]:.2f} m)")
        return chest_bin, info

    # Peaks must be prominent above median
    min_prominence = prominence_factor * std_var

    peaks, properties = signal.find_peaks(
        search_variance,
        prominence=min_prominence,
        distance=10  # At least 10 bins apart
    )

    # Step 4: Select best peak
    if len(peaks) == 0:
        # No prominent peaks found - fall back to maximum in search range
        print("Warning: No prominent peaks found, using maximum")
        chest_bin_local = np.argmax(search_variance)
        chest_bin = search_bins[chest_bin_local]
    else:
        # Select highest peak
        best_peak_idx = np.argmax(search_variance[peaks])
        chest_bin_local = peaks[best_peak_idx]
        chest_bin = search_bins[chest_bin_local]

    # Diagnostic information
    info = {
        'smoothed_variance': smoothed_variance,
        'search_range': (range_bins[search_bins[0]], range_bins[search_bins[-1]]),
        'num_peaks': len(peaks),
        'peak_prominence': properties.get('prominences', []) if isinstance(properties, dict) else [],
        'chest_range': range_bins[chest_bin],
        'variance_at_chest': smoothed_variance[chest_bin],
        'median_variance': median_var
    }

    print(f"Detected chest at bin {chest_bin} ({range_bins[chest_bin]:.2f} m)")
    print(f"  Variance: {smoothed_variance[chest_bin]:.2e} (median: {median_var:.2e})")
    print(f"  Found {len(peaks)} prominent peaks in search range")

    return chest_bin, info


def improved_phase_extraction(chest_signal, sampling_rate,
                              detrend=True, remove_dc=True, normalize=True):
    """
    Improved phase extraction with detrending and drift removal

    Fixes issues:
    - Decaying exponential instead of oscillation
    - DC drift dominates
    - No periodic structure visible

    Args:
        chest_signal: Complex IQ signal from chest bin
        sampling_rate: Sampling rate in Hz (PRF)
        detrend: Remove linear trend
        remove_dc: Remove DC offset
        normalize: Normalize to zero mean, unit variance

    Returns:
        phase_cleaned: Cleaned phase signal ready for filtering
        info: Diagnostic information
    """

    # Step 1: Extract raw phase
    phase_raw = np.angle(chest_signal)

    # Step 2: Unwrap phase
    phase_unwrapped = np.unwrap(phase_raw)

    # Step 3: Remove DC offset (crucial!)
    if remove_dc:
        phase_no_dc = phase_unwrapped - np.mean(phase_unwrapped)
    else:
        phase_no_dc = phase_unwrapped

    # Step 4: Detrend (remove linear drift)
    if detrend:
        # Fit linear trend and remove it
        t = np.arange(len(phase_no_dc))
        coeffs = np.polyfit(t, phase_no_dc, 1)  # Linear fit
        trend = np.polyval(coeffs, t)
        phase_detrended = phase_no_dc - trend
    else:
        phase_detrended = phase_no_dc

    # Step 5: High-pass filter to remove remaining low-frequency drift
    # This is critical for removing drift < 0.1 Hz
    cutoff = 0.05  # Hz
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff < 0.99:
        try:
            b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
            phase_cleaned = signal.filtfilt(b, a, phase_detrended)
        except ValueError as e:
            print(f"Warning: High-pass filter design failed ({e}), using detrended signal")
            phase_cleaned = phase_detrended
        except Exception as e:
            print(f"Warning: Unexpected filter error ({e}), using detrended signal")
            phase_cleaned = phase_detrended
    else:
        phase_cleaned = phase_detrended

    # Step 6: Optional normalization
    if normalize:
        if np.std(phase_cleaned) > 1e-10:
            phase_cleaned = (phase_cleaned - np.mean(phase_cleaned)) / np.std(phase_cleaned)

    # Diagnostic info
    info = {
        'raw_range': (np.min(phase_raw), np.max(phase_raw)),
        'unwrapped_range': (np.min(phase_unwrapped), np.max(phase_unwrapped)),
        'dc_offset': np.mean(phase_unwrapped),
        'linear_trend_slope': coeffs[0] if detrend else 0,
        'cleaned_range': (np.min(phase_cleaned), np.max(phase_cleaned)),
        'cleaned_std': np.std(phase_cleaned)
    }

    print("Phase extraction:")
    print(f"  Raw phase range: [{info['raw_range'][0]:.3f}, {info['raw_range'][1]:.3f}]")
    print(f"  DC offset removed: {info['dc_offset']:.3f} radians")
    if detrend:
        print(f"  Linear trend removed: slope = {info['linear_trend_slope']:.3e}")
    print(f"  Cleaned phase range: [{info['cleaned_range'][0]:.3f}, {info['cleaned_range'][1]:.3f}]")

    return phase_cleaned, info


def improved_clutter_removal(rtm_matrix, method='moving_average', window_size=50):
    """
    Improved clutter removal for better range-time matrix structure

    Fixes issues:
    - Flat, uniform range-time matrix
    - No visible chest reflection
    - Phase is random noise

    Args:
        rtm_matrix: Range-time matrix (slow_time x fast_time)
        method: 'mean', 'median', 'moving_average'
        window_size: Window for moving average

    Returns:
        clutter_removed: Matrix with clutter removed
        clutter_profile: The removed clutter
    """

    if method == 'mean':
        # Per-bin mean subtraction (across slow-time)
        clutter_profile = np.mean(rtm_matrix, axis=0, keepdims=True)
        clutter_removed = rtm_matrix - clutter_profile

    elif method == 'median':
        # More robust to outliers
        clutter_profile = np.median(rtm_matrix, axis=0, keepdims=True)
        clutter_removed = rtm_matrix - clutter_profile

    elif method == 'moving_average':
        # Removes slow drift while preserving breathing
        # Handle complex matrices by filtering real and imaginary parts separately
        if np.iscomplexobj(rtm_matrix):
            clutter_real = uniform_filter1d(rtm_matrix.real, size=window_size, axis=0, mode='nearest')
            clutter_imag = uniform_filter1d(rtm_matrix.imag, size=window_size, axis=0, mode='nearest')
            clutter_estimate = clutter_real + 1j * clutter_imag
        else:
            clutter_estimate = uniform_filter1d(rtm_matrix, size=window_size, axis=0, mode='nearest')

        clutter_removed = rtm_matrix - clutter_estimate
        clutter_profile = clutter_estimate

    else:
        print(f"Unknown method: {method}, using mean")
        clutter_profile = np.mean(rtm_matrix, axis=0, keepdims=True)
        clutter_removed = rtm_matrix - clutter_profile

    print(f"Clutter removal ({method}):")
    print(f"  Input range: [{np.min(np.abs(rtm_matrix)):.3e}, {np.max(np.abs(rtm_matrix)):.3e}]")
    print(f"  Output range: [{np.min(np.abs(clutter_removed)):.3e}, {np.max(np.abs(clutter_removed)):.3e}]")
    print(f"  Mean reduction: {np.mean(np.abs(rtm_matrix)):.3e} → {np.mean(np.abs(clutter_removed)):.3e}")

    return clutter_removed, clutter_profile


def diagnose_signal_quality(chest_signal, sampling_rate):
    """
    Diagnose signal quality issues

    Args:
        chest_signal: Complex IQ from chest bin
        sampling_rate: Sampling rate (PRF)

    Returns:
        report: Dictionary with diagnostic info
    """

    magnitude = np.abs(chest_signal)
    phase = np.angle(chest_signal)

    # Magnitude statistics
    mag_mean = np.mean(magnitude)
    mag_std = np.std(magnitude)
    mag_range = (np.min(magnitude), np.max(magnitude))

    # Phase statistics
    phase_unwrapped = np.unwrap(phase)
    phase_diff = np.diff(phase_unwrapped)
    phase_std = np.std(phase_diff)

    # Temporal characteristics
    from scipy.fft import fft, fftfreq
    N = len(chest_signal)
    phase_fft = fft(phase_unwrapped)
    freqs = fftfreq(N, d=1.0/sampling_rate)

    # Power in breathing band
    positive_idx = freqs > 0
    breathing_idx = (freqs >= Config.BREATHING_FREQ_MIN) & (freqs <= Config.BREATHING_FREQ_MAX)
    breathing_power = np.sum(np.abs(phase_fft[breathing_idx])**2)
    total_power = np.sum(np.abs(phase_fft[positive_idx])**2)

    breathing_fraction = breathing_power / (total_power + 1e-10)

    report = {
        'magnitude': {
            'mean': mag_mean,
            'std': mag_std,
            'range': mag_range,
            'cv': mag_std / (mag_mean + 1e-10)  # Coefficient of variation
        },
        'phase': {
            'unwrapped_range': (np.min(phase_unwrapped), np.max(phase_unwrapped)),
            'diff_std': phase_std,
            'has_modulation': phase_std > 0.01  # Some threshold
        },
        'spectral': {
            'breathing_fraction': breathing_fraction,
            'total_power': total_power,
            'breathing_power': breathing_power,
            'likely_breathing': breathing_fraction > 0.05  # At least 5% in breathing band
        }
    }

    print("\n" + "="*60)
    print("SIGNAL QUALITY DIAGNOSIS")
    print("="*60)
    print(f"Magnitude:")
    print(f"  Mean: {mag_mean:.3e}, Std: {mag_std:.3e}, CV: {report['magnitude']['cv']:.3f}")
    print(f"  Range: [{mag_range[0]:.3e}, {mag_range[1]:.3e}]")
    print(f"\nPhase:")
    print(f"  Unwrapped range: [{report['phase']['unwrapped_range'][0]:.3f}, {report['phase']['unwrapped_range'][1]:.3f}] rad")
    print(f"  Diff std: {phase_std:.3e}")
    print(f"  Has modulation: {report['phase']['has_modulation']}")
    print(f"\nSpectral:")
    print(f"  Total power: {total_power:.3e}")
    print(f"  Breathing power: {breathing_power:.3e} ({breathing_fraction*100:.1f}%)")
    print(f"  Likely breathing signal: {report['spectral']['likely_breathing']}")
    print("="*60 + "\n")

    return report


# Example usage function
def process_with_improvements(rtm_matrix, rtm_object):
    """
    Process data with all improvements

    Args:
        rtm_matrix: Range-time matrix (already constructed)
        rtm_object: RangeTimeMatrix object (for getting range bins)

    Returns:
        chest_bin, chest_signal, phase_cleaned, info
    """

    from preprocessing import Preprocessor
    from respiration_extraction import RespirationExtractor

    print("\n" + "="*70)
    print("PROCESSING WITH IMPROVEMENTS")
    print("="*70)

    # Step 1: Improved clutter removal
    print("\n[1] Clutter Removal...")
    clutter_removed, _ = improved_clutter_removal(rtm_matrix, method='moving_average', window_size=50)

    # Step 2: Calculate variance for chest detection
    print("\n[2] Chest Detection...")
    preprocessor = Preprocessor(clutter_removed)
    preprocessor.calculate_slow_time_variance()

    # Use improved chest detection
    range_bins = rtm_object.get_range_bins()
    chest_bin, chest_info = improved_chest_detection(
        preprocessor.variance_profile,
        range_bins,
        smoothing_sigma=5.0,
        min_range_m=0.3,
        max_range_m=3.0
    )

    # Step 3: Extract chest signal
    print("\n[3] Extracting Chest Signal...")
    chest_signal = clutter_removed[:, chest_bin]

    # Step 4: Diagnose signal quality
    print("\n[4] Signal Quality Check...")
    quality_report = diagnose_signal_quality(chest_signal, Config.PULSE_REPETITION_FREQ)

    # Step 5: Improved phase extraction
    print("\n[5] Phase Extraction...")
    phase_cleaned, phase_info = improved_phase_extraction(
        chest_signal,
        Config.PULSE_REPETITION_FREQ,
        detrend=True,
        remove_dc=True,
        normalize=False  # Let the bandpass filter handle normalization
    )

    # Step 6: Apply bandpass filter (0.1-0.5 Hz)
    print("\n[6] Bandpass Filtering...")
    nyquist = Config.PULSE_REPETITION_FREQ / 2
    low = Config.BREATHING_FREQ_MIN / nyquist
    high = Config.BREATHING_FREQ_MAX / nyquist

    b, a = signal.butter(4, [low, high], btype='band')
    phase_filtered = signal.filtfilt(b, a, phase_cleaned)

    print(f"  Filtered range: [{np.min(phase_filtered):.3f}, {np.max(phase_filtered):.3f}]")
    print(f"  Filtered std: {np.std(phase_filtered):.3f}")

    # Return all useful information
    info = {
        'chest_bin': chest_bin,
        'chest_info': chest_info,
        'quality_report': quality_report,
        'phase_info': phase_info,
        'clutter_removed_matrix': clutter_removed,
        'phase_filtered': phase_filtered
    }

    return chest_signal, info


if __name__ == "__main__":
    print("Processing fixes module loaded.")
    print("Import this module and use:")
    print("  - improved_chest_detection()")
    print("  - improved_phase_extraction()")
    print("  - improved_clutter_removal()")
    print("  - diagnose_signal_quality()")
    print("  - process_with_improvements() [complete workflow]")
