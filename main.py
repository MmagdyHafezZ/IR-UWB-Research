"""
Main Execution Script for IR-UWB Respiration Detection System
Integrates all modules: SDR capture, range-time matrix construction,
preprocessing, respiration extraction, and visualization
"""

import numpy as np
import os
import sys
import argparse
from datetime import datetime

from config import Config
from sdr_capture import SDRCapture
from range_time_matrix import RangeTimeMatrix
from preprocessing import Preprocessor
from respiration_extraction import RespirationExtractor
from visualization import Visualizer

class RespirationDetectionSystem:
    """Main system class integrating all components"""

    def __init__(self, mode='realtime', load_data=None):
        """
        Initialize the system

        Args:
            mode: 'realtime' for SDR capture, 'offline' for processing saved data
            load_data: Path to load previously saved data (for offline mode)
        """
        self.mode = mode
        self.load_data_path = load_data

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{Config.OUTPUT_DIR}/{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        self.sdr_capture = None
        self.raw_data = None
        self.rtm = None
        self.preprocessor = None
        self.extractor = None
        self.visualizer = Visualizer(self.output_dir)

        self.results = {}

    # >>> ADDED: small helper to compute a safe breathing_rate_avg
    @staticmethod
    def _compute_breathing_rate_avg(results_dict):
        """
        Compute a safe breathing_rate_avg from available fields.
        Uses the mean of breathing_rate_time and breathing_rate_freq if both exist,
        otherwise uses whichever is available, or 0.0 as a last resort.
        """
        br_time = results_dict.get('breathing_rate_time')
        br_freq = results_dict.get('breathing_rate_freq')

        valid = [v for v in (br_time, br_freq) if isinstance(v, (int, float, np.floating))]
        if not valid:
            return 0.0
        return float(sum(valid) / len(valid))
    # <<< END ADDED

    def step1_capture_data(self):
        """Step 1: Capture raw IQ data from SDR"""
        print("\n" + "=" * 70)
        print("STEP 1: SDR SIGNAL CAPTURE")
        print("=" * 70)

        if self.mode == 'offline' and self.load_data_path is not None:
            print(f"Loading data from {self.load_data_path}...")
            self.raw_data = np.load(self.load_data_path)
            print(f"Loaded data shape: {self.raw_data.shape}")
        else:
            print("Initializing SDR for real-time capture...")
            self.sdr_capture = SDRCapture()

            try:
                self.sdr_capture.setup_transmitter()
                self.sdr_capture.setup_receiver()
                self.sdr_capture.generate_impulse_signal()
                self.raw_data = self.sdr_capture.record_pulse_sequence()
                print(f"Captured data shape: {self.raw_data.shape}")

                raw_data_path = f"{self.output_dir}/raw_iq_data.npy"
                np.save(raw_data_path, self.raw_data)
                print(f"Saved raw data to {raw_data_path}")

            finally:
                if self.sdr_capture is not None:
                    self.sdr_capture.cleanup()

        self.results['raw_data_shape'] = self.raw_data.shape
        return self.raw_data

    def step2_construct_range_time_matrix(self):
        """Step 2: Construct and align range-time matrix"""
        print("\n" + "=" * 70)
        print("STEP 2: RANGE-TIME MATRIX CONSTRUCTION")
        print("=" * 70)

        self.rtm = RangeTimeMatrix(self.raw_data)

        matrix = self.rtm.construct_matrix()
        aligned_matrix = self.rtm.align_pulses(method="cross_correlation")

        matrix_path = f"{self.output_dir}/range_time_matrix.npy"
        self.rtm.save_matrix(matrix_path)

        range_bins = self.rtm.get_range_bins()
        time_axis = self.rtm.get_time_axis()

        self.results['range_bins'] = range_bins
        self.results['time_axis'] = time_axis

        if Config.PLOT_RESULTS:
            self.visualizer.plot_range_time_matrix(aligned_matrix, range_bins, time_axis)

        return aligned_matrix

    def step3_preprocess_data(self):
        """Step 3: Apply preprocessing pipeline"""
        print("\n" + "=" * 70)
        print("STEP 3: PREPROCESSING PIPELINE")
        print("=" * 70)

        matrix = self.rtm.aligned_matrix if self.rtm.aligned_matrix is not None else self.rtm.range_time_matrix
        self.preprocessor = Preprocessor(matrix)

        processed_matrix, chest_bin = self.preprocessor.run_full_pipeline(
            clutter_method="mean_subtraction",
            apply_highpass=True,
            normalize=True
        )

        processed_path = f"{self.output_dir}/preprocessed_matrix.npy"
        np.save(processed_path, processed_matrix)
        print(f"Saved preprocessed matrix to {processed_path}")

        variance_profile = self.preprocessor.variance_profile
        variance_path = f"{self.output_dir}/variance_profile.npy"
        np.save(variance_path, variance_profile)

        self.results['chest_bin'] = chest_bin
        self.results['chest_range'] = self.results['range_bins'][chest_bin]
        self.results['variance_profile'] = variance_profile

        if Config.PLOT_RESULTS:
            self.visualizer.plot_variance_profile(variance_profile,
                                                 self.results['range_bins'],
                                                 chest_bin)

        return processed_matrix, chest_bin

    def step4_extract_respiration(self, chest_bin):
        """Step 4: Extract respiration signal and estimate breathing rate"""
        print("\n" + "=" * 70)
        print("STEP 4: RESPIRATION EXTRACTION")
        print("=" * 70)

        chest_signal = self.preprocessor.extract_range_bin(chest_bin)

        chest_signal_path = f"{self.output_dir}/chest_signal.npy"
        np.save(chest_signal_path, chest_signal)

        self.extractor = RespirationExtractor(chest_signal, sampling_rate=Config.PULSE_REPETITION_FREQ)

        respiration_results = self.extractor.run_full_analysis()

        # >>> ADDED: ensure breathing_rate_avg exists to avoid KeyError
        if 'breathing_rate_avg' not in respiration_results:
            respiration_results['breathing_rate_avg'] = self._compute_breathing_rate_avg(respiration_results)
        # <<< END ADDED

        breathing_time, breathing_waveform = self.extractor.get_breathing_waveform()
        frequencies, spectrum = self.extractor.get_frequency_spectrum()

        np.save(f"{self.output_dir}/breathing_waveform.npy", breathing_waveform)
        np.save(f"{self.output_dir}/breathing_time.npy", breathing_time)
        np.save(f"{self.output_dir}/frequency_spectrum.npy", spectrum)
        np.save(f"{self.output_dir}/frequencies.npy", frequencies)

        self.results.update(respiration_results)
        self.results['breathing_time'] = breathing_time
        self.results['breathing_waveform'] = breathing_waveform
        self.results['frequencies'] = frequencies
        self.results['spectrum'] = spectrum

        if Config.PLOT_RESULTS:
            br_avg = respiration_results.get('breathing_rate_avg', 0.0)
            br_freq = respiration_results.get('breathing_rate_freq', 0.0)

            self.visualizer.plot_breathing_waveform(
                breathing_time,
                breathing_waveform,
                br_avg
            )
            self.visualizer.plot_frequency_spectrum(
                frequencies,
                spectrum,
                br_freq / 60.0 if br_freq else 0.0
            )

        return respiration_results

    def step5_generate_comprehensive_report(self):
        """Step 5: Generate comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("STEP 5: COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 70)

        if Config.PLOT_RESULTS:
            matrix = self.rtm.aligned_matrix if self.rtm.aligned_matrix is not None else self.rtm.range_time_matrix

            self.visualizer.plot_complete_analysis(
                rtm=matrix,
                variance_profile=self.results['variance_profile'],
                range_bins=self.results['range_bins'],
                time_axis=self.results['time_axis'],
                chest_bin=self.results['chest_bin'],
                breathing_waveform=self.results['breathing_waveform'],
                breathing_time=self.results['breathing_time'],
                frequencies=self.results['frequencies'],
                spectrum=self.results['spectrum'],
                results=self.results
            )

        import json

        # >>> CHANGED: use get() and recompute avg safely
        br_time = float(self.results.get('breathing_rate_time', 0.0))
        br_freq = float(self.results.get('breathing_rate_freq', 0.0))
        br_avg = float(self.results.get('breathing_rate_avg', self._compute_breathing_rate_avg(self.results)))
        signal_quality = float(self.results.get('signal_quality', 0.0))

        pattern = self.results.get('pattern_metrics', {})
        breathing_depth = float(pattern.get('breathing_depth', 0.0))
        regularity = float(pattern.get('regularity', 0.0))
        ie_ratio = float(pattern.get('ie_ratio', 0.0))
        num_breaths = int(pattern.get('num_breaths', 0))
        # <<< END CHANGED

        results_json = {
            'breathing_rate_time_bpm': br_time,
            'breathing_rate_freq_bpm': br_freq,
            'breathing_rate_avg_bpm': br_avg,
            'signal_quality': signal_quality,
            'chest_range_m': float(self.results.get('chest_range', 0.0)),
            'chest_bin': int(self.results.get('chest_bin', 0)),
            'pattern_metrics': {
                'breathing_depth': breathing_depth,
                'regularity': regularity,
                'ie_ratio': ie_ratio,
                'num_breaths': num_breaths
            },
            'system_config': {
                'tx_freq_ghz': Config.TX_FREQ / 1e9,
                'rx_freq_ghz': Config.RX_FREQ / 1e9,
                'prf_hz': Config.PULSE_REPETITION_FREQ,
                'range_resolution_m': Config.get_range_resolution(),
                'max_range_m': Config.get_max_range()
            }
        }

        results_path = f"{self.output_dir}/analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=4)

        print(f"Saved analysis results to {results_path}")

        self.print_summary()

    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 70)
        print("RESPIRATION DETECTION SUMMARY")
        print("=" * 70)

        # >>> CHANGED: use safe .get() and recompute avg
        br_time = float(self.results.get('breathing_rate_time', 0.0))
        br_freq = float(self.results.get('breathing_rate_freq', 0.0))
        br_avg = float(self.results.get('breathing_rate_avg', self._compute_breathing_rate_avg(self.results)))
        signal_quality = float(self.results.get('signal_quality', 0.0))
        chest_range = float(self.results.get('chest_range', 0.0))
        chest_bin = int(self.results.get('chest_bin', 0))

        pattern = self.results.get('pattern_metrics', {})
        num_breaths = int(pattern.get('num_breaths', 0))
        breathing_depth = float(pattern.get('breathing_depth', 0.0))
        regularity = float(pattern.get('regularity', 0.0))
        ie_ratio = float(pattern.get('ie_ratio', 0.0))

        print(f"Breathing Rate (Time-domain): {br_time:.1f} BPM")
        print(f"Breathing Rate (Freq-domain): {br_freq:.1f} BPM")
        print(f"Average Breathing Rate: {br_avg:.1f} BPM")
        print(f"\nSignal Quality: {signal_quality:.2f}")
        print(f"\nChest Location: {chest_range:.2f} m (bin {chest_bin})")
        print(f"\nBreathing Pattern:")
        print(f"  Number of Breaths: {num_breaths}")
        print(f"  Breathing Depth: {breathing_depth:.3f}")
        print(f"  Regularity: {regularity:.2f}")
        print(f"  I/E Ratio: {ie_ratio:.2f}")
        print("=" * 70)
        print(f"\nAll results saved to: {self.output_dir}")
        print("=" * 70)
        # <<< END CHANGED

    def run_complete_pipeline(self):
        """Run the complete detection pipeline"""
        print("\n" + "=" * 70)
        print("IR-UWB RESPIRATION DETECTION SYSTEM")
        print("=" * 70)
        Config.print_config()

        try:
            self.step1_capture_data()
            self.step2_construct_range_time_matrix()
            processed_matrix, chest_bin = self.step3_preprocess_data()
            self.step4_extract_respiration(chest_bin)
            self.step5_generate_comprehensive_report()

            print("\n" + "=" * 70)
            print("PIPELINE EXECUTION COMPLETE")
            print("=" * 70)

            return self.results

        except Exception as e:
            print(f"\nError during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='IR-UWB Respiration Detection System')
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'offline'],
                       help='Operation mode: realtime (SDR capture) or offline (load saved data)')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load saved raw data (for offline mode)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')

    args = parser.parse_args()

    if args.no_plot:
        Config.PLOT_RESULTS = False

    system = RespirationDetectionSystem(mode=args.mode, load_data=args.load)
    results = system.run_complete_pipeline()

    if results is not None:
        print("\nSystem execution successful!")
        return 0
    else:
        print("\nSystem execution failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
