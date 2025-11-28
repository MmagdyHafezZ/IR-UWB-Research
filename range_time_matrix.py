"""
Range-Time Matrix Construction Module
Transforms captured IQ sequences into 2D range-time matrix
Handles pulse alignment and timing jitter compensation
"""

import numpy as np
from config import Config
import numpy as np
from scipy import signal
if not hasattr(signal, "find_peaks"):
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

    signal.find_peaks = _fallback_find_peaks

class RangeTimeMatrix:
    """Constructs and manages the range-time matrix"""

    def __init__(self, raw_data):
        """
        Initialize with raw IQ data

        Args:
            raw_data: 2D array [num_pulses, samples_per_pulse] or 1D continuous array
        """
        self.raw_data = raw_data
        self.range_time_matrix = None
        self.aligned_matrix = None

    def extract_pulses_from_continuous(self, continuous_data):
        """
        Extract individual pulses from continuous data stream

        Args:
            continuous_data: 1D array of continuous IQ samples

        Returns:
            2D array [num_pulses, samples_per_pulse]
        """
        print("Extracting pulses from continuous data...")

        
        samples_per_interval = int(Config.RX_SAMPLE_RATE / Config.PULSE_REPETITION_FREQ)

        
        envelope = np.abs(continuous_data)

        
        
        min_distance = int(0.8 * samples_per_interval)  

        
        peaks, properties = signal.find_peaks(envelope,
                                             distance=min_distance,
                                             height=np.mean(envelope) * 2)

        print(f"Detected {len(peaks)} pulses")

        
        pulses = []
        samples_per_pulse = Config.SAMPLES_PER_PULSE

        for peak_idx in peaks:
            
            start_idx = peak_idx
            end_idx = start_idx + samples_per_pulse

            if end_idx <= len(continuous_data):
                pulse_data = continuous_data[start_idx:end_idx]
                pulses.append(pulse_data)

        
        pulse_matrix = np.array(pulses)
        print(f"Extracted pulse matrix shape: {pulse_matrix.shape}")

        return pulse_matrix

    def construct_matrix(self):
        """
        Construct 2D range-time matrix

        Dimensions:
        - Fast-time (columns): Range bins (samples within a pulse)
        - Slow-time (rows): Frames (pulse number)

        Returns:
            2D complex array [slow_time, fast_time]
        """
        print("Constructing range-time matrix...")

        
        if self.raw_data.ndim == 1:
            self.range_time_matrix = self.extract_pulses_from_continuous(self.raw_data)
        else:
            
            self.range_time_matrix = self.raw_data

        print(f"Range-Time Matrix shape: {self.range_time_matrix.shape}")
        print(f"  Fast-time (range bins): {self.range_time_matrix.shape[1]}")
        print(f"  Slow-time (frames): {self.range_time_matrix.shape[0]}")

        return self.range_time_matrix

    def align_pulses(self, method="cross_correlation"):
        """
        Align pulses to compensate for timing jitter between frames

        Args:
            method: Alignment method ('cross_correlation' or 'peak_alignment')

        Returns:
            Aligned range-time matrix
        """
        print(f"Aligning pulses using {method}...")

        if self.range_time_matrix is None:
            self.construct_matrix()

        num_pulses, num_samples = self.range_time_matrix.shape

        if method == "cross_correlation":
            
            reference_pulse = self.range_time_matrix[0, :]
            aligned_matrix = np.zeros_like(self.range_time_matrix)
            aligned_matrix[0, :] = reference_pulse

            max_shift = 50  

            for i in range(1, num_pulses):
                current_pulse = self.range_time_matrix[i, :]

                
                correlation = signal.correlate(current_pulse, reference_pulse, mode='same')

                
                peak_idx = np.argmax(np.abs(correlation))
                shift = peak_idx - num_samples // 2

                
                shift = np.clip(shift, -max_shift, max_shift)

                
                if shift > 0:
                    
                    aligned_matrix[i, shift:] = current_pulse[:-shift] if shift < num_samples else 0
                elif shift < 0:
                    
                    
                    end_pos = min(num_samples, num_samples + shift)  
                    aligned_matrix[i, -shift:] = current_pulse[:end_pos]
                else:
                    aligned_matrix[i, :] = current_pulse

                if (i + 1) % 100 == 0:
                    print(f"  Aligned {i + 1}/{num_pulses} pulses")

        elif method == "peak_alignment":
            
            aligned_matrix = np.zeros_like(self.range_time_matrix)

            
            peak_locations = np.argmax(np.abs(self.range_time_matrix), axis=1)
            reference_peak = np.median(peak_locations).astype(int)

            for i in range(num_pulses):
                current_peak = peak_locations[i]
                shift = current_peak - reference_peak

                
                current_pulse = self.range_time_matrix[i, :]
                if shift > 0:
                    aligned_matrix[i, shift:] = current_pulse[:-shift]
                elif shift < 0:
                    aligned_matrix[i, :shift] = current_pulse[-shift:]
                else:
                    aligned_matrix[i, :] = current_pulse

        else:
            print(f"Unknown alignment method: {method}")
            aligned_matrix = self.range_time_matrix

        self.aligned_matrix = aligned_matrix
        print("Pulse alignment complete")

        return self.aligned_matrix

    def get_range_bins(self):
        """
        Get range values for each fast-time bin

        Returns:
            Array of range values in meters
        """
        num_bins = self.range_time_matrix.shape[1]
        sample_time = 1.0 / Config.RX_SAMPLE_RATE
        time_bins = np.arange(num_bins) * sample_time
        range_bins = (Config.C * time_bins) / 2  

        return range_bins

    def get_time_axis(self):
        """
        Get time values for slow-time axis

        Returns:
            Array of time values in seconds
        """
        num_frames = self.range_time_matrix.shape[0]
        time_axis = np.arange(num_frames) * Config.get_pulse_interval()

        return time_axis

    def get_matrix_magnitude(self):
        """Get magnitude of the range-time matrix"""
        if self.aligned_matrix is not None:
            return np.abs(self.aligned_matrix)
        elif self.range_time_matrix is not None:
            return np.abs(self.range_time_matrix)
        else:
            return None

    def get_matrix_phase(self):
        """Get phase of the range-time matrix"""
        if self.aligned_matrix is not None:
            return np.angle(self.aligned_matrix)
        elif self.range_time_matrix is not None:
            return np.angle(self.range_time_matrix)
        else:
            return None

    def save_matrix(self, filename):
        """Save range-time matrix to file"""
        matrix_to_save = self.aligned_matrix if self.aligned_matrix is not None else self.range_time_matrix

        if matrix_to_save is not None:
            np.save(filename, matrix_to_save)
            print(f"Saved range-time matrix to {filename}")
        else:
            print("No matrix to save")

    def load_matrix(self, filename):
        """Load range-time matrix from file"""
        self.range_time_matrix = np.load(filename)
        print(f"Loaded range-time matrix from {filename}")
        print(f"Matrix shape: {self.range_time_matrix.shape}")

        return self.range_time_matrix


def test_range_time_matrix():
    """Test function for range-time matrix construction"""
    print("Testing Range-Time Matrix Module")
    print("=" * 60)

    
    print("Generating synthetic pulse data...")

    num_pulses = Config.NUM_PULSES
    samples_per_pulse = Config.SAMPLES_PER_PULSE

    
    target_range_bin = 200
    target_amplitude = 1.0

    synthetic_data = np.zeros((num_pulses, samples_per_pulse), dtype=np.complex64)

    for i in range(num_pulses):
        
        phase = 2 * np.pi * 0.2 * np.sin(2 * np.pi * 0.3 * i / num_pulses)  
        synthetic_data[i, target_range_bin] = target_amplitude * np.exp(1j * phase)

        
        noise = 0.1 * (np.random.randn(samples_per_pulse) + 1j * np.random.randn(samples_per_pulse))
        synthetic_data[i, :] += noise

        
        jitter = np.random.randint(-3, 4)
        if jitter != 0:
            synthetic_data[i, :] = np.roll(synthetic_data[i, :], jitter)

    
    rtm = RangeTimeMatrix(synthetic_data)
    matrix = rtm.construct_matrix()

    print(f"\nConstructed matrix shape: {matrix.shape}")

    
    aligned = rtm.align_pulses(method="cross_correlation")
    print(f"Aligned matrix shape: {aligned.shape}")

    
    range_bins = rtm.get_range_bins()
    time_axis = rtm.get_time_axis()

    print(f"\nRange bins: {range_bins[0]:.2f} to {range_bins[-1]:.2f} m")
    print(f"Time axis: {time_axis[0]:.2f} to {time_axis[-1]:.2f} s")

    
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    rtm.save_matrix(f"{Config.OUTPUT_DIR}/range_time_matrix.npy")


if __name__ == "__main__":
    Config.print_config()
    test_range_time_matrix()
