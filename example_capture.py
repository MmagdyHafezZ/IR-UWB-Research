#!/usr/bin/env python3
"""
Example: Capture Data from SDR
Simple example showing how to use the SDR capture interface
"""

import numpy as np
import os
from config import Config
from sdr_capture import SDRCapture

def capture_with_hardware():
    """Capture data using actual SDR hardware"""
    print("=" * 70)
    print("SDR DATA CAPTURE - Hardware Mode")
    print("=" * 70)

    try:
        # Initialize SDR (will raise error if SoapySDR not available)
        print("\n[1] Initializing SDR hardware...")
        sdr = SDRCapture(hardware_mode=True)

        # Setup transmitter
        print("\n[2] Setting up transmitter...")
        sdr.setup_transmitter()

        # Setup receiver
        print("\n[3] Setting up receiver...")
        sdr.setup_receiver()

        # Generate impulse signal
        print("\n[4] Generating impulse signal...")
        sdr.generate_impulse_signal()

        # Capture data
        print(f"\n[5] Capturing {Config.NUM_PULSES} pulses...")
        print(f"    Duration: {Config.NUM_PULSES / Config.PULSE_REPETITION_FREQ:.1f} seconds")

        raw_data = sdr.record_pulse_sequence()

        print(f"\n✓ Capture complete!")
        print(f"  Data shape: {raw_data.shape}")
        print(f"  Data type: {raw_data.dtype}")

        # Save data
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        output_file = f"{Config.OUTPUT_DIR}/raw_data.npy"
        np.save(output_file, raw_data)
        print(f"  Saved to: {output_file}")

        return raw_data

    except RuntimeError as e:
        print(f"\n✗ Hardware error: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: python3 diagnose_sdr.py")
        print("  2. Check: HARDWARE_SETUP.md")
        print("  3. Verify SoapySDR is installed")
        return None

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def capture_synthetic():
    """Generate synthetic data for testing (no hardware required)"""
    print("=" * 70)
    print("SDR DATA CAPTURE - Synthetic Mode (No Hardware)")
    print("=" * 70)

    print("\n[1] Generating synthetic radar data...")

    # Parameters
    num_pulses = Config.NUM_PULSES
    samples_per_pulse = Config.SAMPLES_PER_PULSE
    target_bin = 256  # Chest location
    breathing_rate_bpm = 15

    # Create synthetic data with breathing signal
    data = np.zeros((num_pulses, samples_per_pulse), dtype=np.complex64)

    # Static clutter
    np.random.seed(42)
    for i in range(samples_per_pulse):
        clutter_amplitude = 0.2
        clutter_phase = np.random.rand() * 2 * np.pi
        data[:, i] = clutter_amplitude * np.exp(1j * clutter_phase)

    # Breathing signal
    time = np.arange(num_pulses) / Config.PULSE_REPETITION_FREQ
    breathing_freq = breathing_rate_bpm / 60

    # Realistic chest displacement (6mm peak-to-peak)
    displacement_mm = 6.0 * np.sin(2 * np.pi * breathing_freq * time)

    # Convert to phase at 2.45 GHz
    wavelength_mm = 122.0  # c/f = 3e8 / 2.45e9
    phase_mod = (4 * np.pi / wavelength_mm) * displacement_mm

    # Add chest reflection
    chest_amplitude = 5.0
    for i in range(num_pulses):
        data[i, target_bin] = chest_amplitude * np.exp(1j * phase_mod[i])

    # Noise
    data += 0.02 * (np.random.randn(num_pulses, samples_per_pulse) +
                    1j * np.random.randn(num_pulses, samples_per_pulse))

    print(f"✓ Generated synthetic data")
    print(f"  Shape: {data.shape}")
    print(f"  Expected breathing rate: {breathing_rate_bpm} BPM")

    # Save data
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_file = f"{Config.OUTPUT_DIR}/synthetic_data.npy"
    np.save(output_file, data)
    print(f"  Saved to: {output_file}")

    return data


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("IR-UWB RESPIRATION DETECTION - DATA CAPTURE")
    print("=" * 70)

    # Check if SoapySDR is available
    try:
        import SoapySDR
        soapy_available = True
        print("\n✓ SoapySDR is installed")
    except ImportError:
        soapy_available = False
        print("\n⚠ SoapySDR not installed")

    # Decide mode
    if soapy_available:
        print("\nOptions:")
        print("  1. Capture from hardware (requires SDR connected)")
        print("  2. Generate synthetic data (no hardware needed)")
        choice = input("\nSelect option (1 or 2, default=2): ").strip()

        if choice == "1":
            print("\nAttempting hardware capture...")
            raw_data = capture_with_hardware()
            if raw_data is None:
                print("\nHardware capture failed. Falling back to synthetic data.")
                raw_data = capture_synthetic()
        else:
            raw_data = capture_synthetic()
    else:
        print("\nUsing synthetic data (SoapySDR not installed)")
        print("To use hardware: pip install SoapySDR (see HARDWARE_SETUP.md)")
        raw_data = capture_synthetic()

    # Next steps
    if raw_data is not None:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Process the captured data:")
        print("   python3 -c \"")
        print("   from range_time_matrix import RangeTimeMatrix")
        print("   from preprocessing import Preprocessor")
        print("   from respiration_extraction import RespirationExtractor")
        print("   import numpy as np")
        print(f"   data = np.load('{Config.OUTPUT_DIR}/raw_data.npy' if hardware else '{Config.OUTPUT_DIR}/synthetic_data.npy')")
        print("   rtm = RangeTimeMatrix(data)")
        print("   matrix = rtm.construct_matrix()")
        print("   prep = Preprocessor(matrix)")
        print("   prep.remove_clutter_mean_subtraction()")
        print("   chest_signal = prep.extract_range_bin(256)")
        print("   extractor = RespirationExtractor(chest_signal)")
        print("   results = extractor.run_full_analysis()")
        print("   print(f'Breathing Rate: {results[\\\"baseline\\\"][\\\"breathing_rate_avg\\\"]} BPM')")
        print("   \"")
        print("\n2. Or run complete pipeline:")
        print("   python3 main.py")
        print("\n3. View documentation:")
        print("   cat SYSTEM_DOCUMENTATION.md")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
