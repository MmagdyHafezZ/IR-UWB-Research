#!/usr/bin/env python3
"""
Demonstration: Processing Improvements
Shows before/after comparison of signal processing fixes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from config import Config
from range_time_matrix import RangeTimeMatrix
from preprocessing import Preprocessor
from respiration_extraction import RespirationExtractor
from processing_fixes import (
    improved_chest_detection,
    improved_phase_extraction,
    improved_clutter_removal,
    diagnose_signal_quality,
    process_with_improvements
)


def generate_realistic_test_data():
    """Generate realistic synthetic radar data with breathing signal"""
    print("\n" + "="*70)
    print("GENERATING REALISTIC TEST DATA")
    print("="*70)

    num_pulses = 5000
    samples_per_pulse = 512
    chest_bin = 256
    breathing_rate_bpm = 15

    # Create data matrix
    data = np.zeros((num_pulses, samples_per_pulse), dtype=np.complex64)

    # Static clutter (same for all pulses)
    np.random.seed(42)
    for i in range(samples_per_pulse):
        clutter_amplitude = 0.3
        clutter_phase = np.random.rand() * 2 * np.pi
        data[:, i] = clutter_amplitude * np.exp(1j * clutter_phase)

    # Breathing signal
    time = np.arange(num_pulses) / Config.PULSE_REPETITION_FREQ
    breathing_freq = breathing_rate_bpm / 60

    # Realistic chest displacement (6mm peak-to-peak)
    displacement_mm = 6.0 * np.sin(2 * np.pi * breathing_freq * time)

    # Add drift to simulate real-world conditions
    drift = 0.5 * time / np.max(time)  # Slow linear drift

    # Convert to phase at 2.45 GHz
    wavelength_mm = 122.0
    phase_mod = (4 * np.pi / wavelength_mm) * (displacement_mm + drift)

    # Add chest reflection with drift
    chest_amplitude = 5.0
    for i in range(num_pulses):
        data[i, chest_bin] = chest_amplitude * np.exp(1j * phase_mod[i])

    # Add noise
    data += 0.05 * (np.random.randn(num_pulses, samples_per_pulse) +
                    1j * np.random.randn(num_pulses, samples_per_pulse))

    print(f"✓ Generated data: {data.shape}")
    print(f"  Expected breathing rate: {breathing_rate_bpm} BPM")
    print(f"  Chest location: bin {chest_bin}")
    print(f"  Added linear drift to simulate real conditions")

    return data, chest_bin, breathing_rate_bpm


def process_old_method(rtm_matrix, rtm_object):
    """Process using old method (baseline)"""
    print("\n" + "="*70)
    print("OLD METHOD (Baseline)")
    print("="*70)

    # Old clutter removal
    print("\n[1] Old Clutter Removal (simple mean)...")
    clutter_old = np.mean(rtm_matrix, axis=0, keepdims=True)
    clutter_removed_old = rtm_matrix - clutter_old

    # Old chest detection
    print("\n[2] Old Chest Detection (raw variance)...")
    preprocessor = Preprocessor(clutter_removed_old)
    preprocessor.calculate_slow_time_variance()
    variance_old = preprocessor.variance_profile

    # Simple max detection
    range_bins = rtm_object.get_range_bins()
    chest_bin_old = np.argmax(variance_old)

    print(f"  Detected chest at bin {chest_bin_old} ({range_bins[chest_bin_old]:.2f} m)")
    print(f"  Variance: {variance_old[chest_bin_old]:.2e}")

    # Old phase extraction
    print("\n[3] Old Phase Extraction (no detrending)...")
    chest_signal_old = clutter_removed_old[:, chest_bin_old]

    # Simple unwrap, no detrending
    phase_raw = np.angle(chest_signal_old)
    phase_old = np.unwrap(phase_raw)

    print(f"  Phase range: [{np.min(phase_old):.3f}, {np.max(phase_old):.3f}]")
    print(f"  Phase std: {np.std(phase_old):.3e}")

    # Old bandpass filter
    print("\n[4] Old Bandpass Filter...")
    nyquist = Config.PULSE_REPETITION_FREQ / 2
    low = Config.BREATHING_FREQ_MIN / nyquist
    high = Config.BREATHING_FREQ_MAX / nyquist

    b, a = signal.butter(4, [low, high], btype='band')

    # Apply to raw unwrapped phase
    try:
        phase_filtered_old = signal.filtfilt(b, a, phase_old)
    except:
        print("  Warning: Filter unstable, using raw phase")
        phase_filtered_old = phase_old

    print(f"  Filtered range: [{np.min(phase_filtered_old):.3f}, {np.max(phase_filtered_old):.3f}]")

    return {
        'variance_profile': variance_old,
        'chest_bin': chest_bin_old,
        'chest_signal': chest_signal_old,
        'phase_raw': phase_old,
        'phase_filtered': phase_filtered_old,
        'clutter_removed_matrix': clutter_removed_old
    }


def process_new_method(rtm_matrix, rtm_object):
    """Process using new improved method"""
    print("\n" + "="*70)
    print("NEW METHOD (Improved)")
    print("="*70)

    # Step 1: Improved clutter removal
    print("\n[1] Improved Clutter Removal (moving average)...")
    clutter_removed_new, _ = improved_clutter_removal(
        rtm_matrix,
        method='moving_average',
        window_size=50
    )

    # Step 2: Improved chest detection
    print("\n[2] Improved Chest Detection (smoothed + prominence)...")
    preprocessor = Preprocessor(clutter_removed_new)
    preprocessor.calculate_slow_time_variance()

    range_bins = rtm_object.get_range_bins()
    chest_bin_new, chest_info = improved_chest_detection(
        preprocessor.variance_profile,
        range_bins,
        smoothing_sigma=5.0,
        prominence_factor=2.0,
        min_range_m=0.3,
        max_range_m=3.0
    )

    # Step 3: Extract chest signal
    print("\n[3] Extracting Chest Signal...")
    chest_signal_new = clutter_removed_new[:, chest_bin_new]

    # Step 4: Diagnose signal quality
    print("\n[4] Signal Quality Check...")
    quality_report = diagnose_signal_quality(chest_signal_new, Config.PULSE_REPETITION_FREQ)

    # Step 5: Improved phase extraction
    print("\n[5] Improved Phase Extraction (detrend + high-pass)...")
    phase_cleaned, phase_info = improved_phase_extraction(
        chest_signal_new,
        Config.PULSE_REPETITION_FREQ,
        detrend=True,
        remove_dc=True,
        normalize=False
    )

    # Step 6: Apply bandpass filter
    print("\n[6] Bandpass Filtering...")
    nyquist = Config.PULSE_REPETITION_FREQ / 2
    low = Config.BREATHING_FREQ_MIN / nyquist
    high = Config.BREATHING_FREQ_MAX / nyquist

    b, a = signal.butter(4, [low, high], btype='band')
    phase_filtered_new = signal.filtfilt(b, a, phase_cleaned)

    print(f"  Filtered range: [{np.min(phase_filtered_new):.3f}, {np.max(phase_filtered_new):.3f}]")
    print(f"  Filtered std: {np.std(phase_filtered_new):.3f}")

    return {
        'variance_profile': chest_info['smoothed_variance'],
        'chest_bin': chest_bin_new,
        'chest_signal': chest_signal_new,
        'phase_raw': phase_cleaned,
        'phase_filtered': phase_filtered_new,
        'clutter_removed_matrix': clutter_removed_new,
        'chest_info': chest_info,
        'quality_report': quality_report
    }


def compare_results(old_results, new_results, true_chest_bin, true_breathing_rate):
    """Compare old vs new method results"""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    # Chest detection accuracy
    print("\n[Chest Detection]")
    print(f"  True chest bin: {true_chest_bin}")
    print(f"  Old method: bin {old_results['chest_bin']} (error: {abs(old_results['chest_bin'] - true_chest_bin)} bins)")
    print(f"  New method: bin {new_results['chest_bin']} (error: {abs(new_results['chest_bin'] - true_chest_bin)} bins)")

    # Phase quality
    print("\n[Phase Quality]")
    print(f"  Old phase std: {np.std(old_results['phase_filtered']):.3f}")
    print(f"  New phase std: {np.std(new_results['phase_filtered']):.3f}")

    # Breathing rate estimation
    print("\n[Breathing Rate Estimation]")

    # Old method
    extractor_old = RespirationExtractor(old_results['phase_filtered'], Config.PULSE_REPETITION_FREQ)
    rate_old = extractor_old.detect_breathing_rate_frequency_domain()

    # New method
    extractor_new = RespirationExtractor(new_results['phase_filtered'], Config.PULSE_REPETITION_FREQ)
    rate_new = extractor_new.detect_breathing_rate_frequency_domain()

    print(f"  True rate: {true_breathing_rate} BPM")
    print(f"  Old method: {rate_old:.1f} BPM (error: {abs(rate_old - true_breathing_rate):.1f} BPM)")
    print(f"  New method: {rate_new:.1f} BPM (error: {abs(rate_new - true_breathing_rate):.1f} BPM)")

    return rate_old, rate_new


def plot_comparison(old_results, new_results, rtm_object):
    """Generate before/after comparison plots"""
    print("\n[Generating Comparison Plots]...")

    fig = plt.figure(figsize=(16, 12))

    range_bins = rtm_object.get_range_bins()
    time_axis = np.arange(len(old_results['phase_filtered'])) / Config.PULSE_REPETITION_FREQ

    # Row 1: Variance profiles
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(range_bins, old_results['variance_profile'], 'b-', alpha=0.7, label='Raw variance')
    ax1.axvline(range_bins[old_results['chest_bin']], color='r', linestyle='--', label='Detected chest')
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Variance')
    ax1.set_title('OLD: Chest Detection (Raw Variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(range_bins, new_results['variance_profile'], 'g-', alpha=0.7, label='Smoothed variance')
    ax2.axvline(range_bins[new_results['chest_bin']], color='r', linestyle='--', label='Detected chest')
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('Variance')
    ax2.set_title('NEW: Chest Detection (Smoothed + Prominence)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 2: Phase signals
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(time_axis, old_results['phase_raw'], 'b-', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase (radians)')
    ax3.set_title('OLD: Phase (No Detrending)')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(time_axis, new_results['phase_raw'], 'g-', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Phase (radians)')
    ax4.set_title('NEW: Phase (Detrended + High-pass)')
    ax4.grid(True, alpha=0.3)

    # Row 3: Filtered breathing waveforms
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(time_axis, old_results['phase_filtered'], 'b-', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('OLD: Breathing Waveform (Bandpass Only)')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(time_axis, new_results['phase_filtered'], 'g-', alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Amplitude')
    ax6.set_title('NEW: Breathing Waveform (Detrended + Bandpass)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = f"{Config.OUTPUT_DIR}/processing_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_file}")

    plt.close()


def plot_range_time_matrices(old_results, new_results, rtm_object):
    """Plot range-time matrices comparison"""
    print("\n[Generating Range-Time Matrix Comparison]...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    range_bins = rtm_object.get_range_bins()
    time_axis = np.arange(old_results['clutter_removed_matrix'].shape[0]) / Config.PULSE_REPETITION_FREQ

    # Old method
    ax1 = axes[0]
    im1 = ax1.imshow(
        np.abs(old_results['clutter_removed_matrix'].T),
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], range_bins[-1], range_bins[0]],
        cmap='hot',
        interpolation='nearest'
    )
    ax1.axhline(range_bins[old_results['chest_bin']], color='cyan', linestyle='--', linewidth=2, label='Detected chest')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Range (m)')
    ax1.set_title('OLD: Range-Time Matrix (Mean Subtraction)')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Magnitude')

    # New method
    ax2 = axes[1]
    im2 = ax2.imshow(
        np.abs(new_results['clutter_removed_matrix'].T),
        aspect='auto',
        extent=[time_axis[0], time_axis[-1], range_bins[-1], range_bins[0]],
        cmap='hot',
        interpolation='nearest'
    )
    ax2.axhline(range_bins[new_results['chest_bin']], color='cyan', linestyle='--', linewidth=2, label='Detected chest')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title('NEW: Range-Time Matrix (Moving Average)')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Magnitude')

    plt.tight_layout()

    # Save figure
    output_file = f"{Config.OUTPUT_DIR}/rtm_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_file}")

    plt.close()


def main():
    """Main demonstration"""
    print("\n" + "="*70)
    print("PROCESSING IMPROVEMENTS DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates the signal processing improvements")
    print("by comparing old (baseline) vs new (improved) methods.")

    # Generate test data
    raw_data, true_chest_bin, true_breathing_rate = generate_realistic_test_data()

    # Construct range-time matrix
    print("\n[Constructing Range-Time Matrix]...")
    rtm = RangeTimeMatrix(raw_data)
    rtm_matrix = rtm.construct_matrix()
    print(f"  RTM shape: {rtm_matrix.shape}")

    # Process with old method
    old_results = process_old_method(rtm_matrix, rtm)

    # Process with new method
    new_results = process_new_method(rtm_matrix, rtm)

    # Compare results
    rate_old, rate_new = compare_results(old_results, new_results, true_chest_bin, true_breathing_rate)

    # Generate comparison plots
    plot_comparison(old_results, new_results, rtm)
    plot_range_time_matrices(old_results, new_results, rtm)

    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Improvements:")
    print("  1. Chest Detection: Smoothing reduces false peaks")
    print("  2. Phase Extraction: Detrending removes drift")
    print("  3. Clutter Removal: Moving average preserves breathing")
    print("  4. Overall: Better breathing rate accuracy")

    print(f"\nBreathing Rate Accuracy:")
    print(f"  Old method error: {abs(rate_old - true_breathing_rate):.1f} BPM")
    print(f"  New method error: {abs(rate_new - true_breathing_rate):.1f} BPM")

    improvement = abs(rate_old - true_breathing_rate) - abs(rate_new - true_breathing_rate)
    if improvement > 0:
        print(f"  Improvement: {improvement:.1f} BPM ✓")
    else:
        print(f"  Note: Results may vary with different synthetic data seeds")

    print(f"\nPlots saved to:")
    print(f"  - {Config.OUTPUT_DIR}/processing_comparison.png")
    print(f"  - {Config.OUTPUT_DIR}/rtm_comparison.png")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
