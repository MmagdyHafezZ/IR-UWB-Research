"""
Preprocessing Pipeline Module
Handles clutter removal, filtering, and signal conditioning
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from config import Config


class Preprocessor:
    """Preprocessing operations for range-time matrix"""

    def __init__(self, range_time_matrix):
        """
        Initialize preprocessor with range-time matrix

        Args:
            range_time_matrix: 2D complex array [slow_time, fast_time]
        """
        self.original_matrix = range_time_matrix
        self.processed_matrix = range_time_matrix.copy()
        self.clutter_removed_matrix = None
        self.filtered_matrix = None
        self.variance_profile = None

    def remove_clutter_mean_subtraction(self):
        """
        Remove static clutter using per-bin mean subtraction

        This removes stationary objects by subtracting the mean across slow-time
        for each range bin, leaving only the time-varying components (breathing)
        """
        print("Removing clutter using per-bin mean subtraction...")

        
        mean_profile = np.mean(self.processed_matrix, axis=0, keepdims=True)

        
        self.clutter_removed_matrix = self.processed_matrix - mean_profile

        print(f"Clutter removal complete. Matrix shape: {self.clutter_removed_matrix.shape}")

        self.processed_matrix = self.clutter_removed_matrix

        return self.clutter_removed_matrix

    def remove_clutter_median_subtraction(self):
        """
        Remove static clutter using per-bin median subtraction

        More robust to outliers than mean subtraction
        """
        print("Removing clutter using per-bin median subtraction...")

        
        median_profile = np.median(self.processed_matrix, axis=0, keepdims=True)

        
        self.clutter_removed_matrix = self.processed_matrix - median_profile

        print(f"Clutter removal complete. Matrix shape: {self.clutter_removed_matrix.shape}")

        self.processed_matrix = self.clutter_removed_matrix

        return self.clutter_removed_matrix

    def remove_clutter_moving_average(self, window_size=50):
        """
        Remove clutter using moving average filter

        Args:
            window_size: Size of moving average window

        This is useful for removing slow drifts while preserving breathing signal
        """
        print(f"Removing clutter using moving average (window={window_size})...")

        
        clutter_estimate = uniform_filter1d(self.processed_matrix, size=window_size, axis=0, mode='nearest')

        
        self.clutter_removed_matrix = self.processed_matrix - clutter_estimate

        print(f"Clutter removal complete. Matrix shape: {self.clutter_removed_matrix.shape}")

        self.processed_matrix = self.clutter_removed_matrix

        return self.clutter_removed_matrix

    def apply_highpass_filter(self, cutoff_freq=None, order=4):
        """
        Apply slow-time high-pass filter for detrending

        Args:
            cutoff_freq: Cutoff frequency in Hz (default from config)
            order: Filter order

        This removes low-frequency drift and leaves respiratory motion
        """
        if cutoff_freq is None:
            cutoff_freq = Config.HIGHPASS_CUTOFF

        print(f"Applying high-pass filter (cutoff={cutoff_freq} Hz, order={order})...")

        
        fs_slow = Config.PULSE_REPETITION_FREQ

        
        nyquist = fs_slow / 2
        normalized_cutoff = cutoff_freq / nyquist

        if normalized_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency {cutoff_freq} Hz is above Nyquist frequency {nyquist} Hz")
            return self.processed_matrix

        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)

        
        filtered_matrix = np.zeros_like(self.processed_matrix)

        for range_bin in range(self.processed_matrix.shape[1]):
            
            real_filtered = signal.filtfilt(b, a, self.processed_matrix[:, range_bin].real)
            imag_filtered = signal.filtfilt(b, a, self.processed_matrix[:, range_bin].imag)
            filtered_matrix[:, range_bin] = real_filtered + 1j * imag_filtered

        self.filtered_matrix = filtered_matrix
        self.processed_matrix = filtered_matrix

        print(f"High-pass filtering complete. Matrix shape: {self.processed_matrix.shape}")

        return self.filtered_matrix

    def calculate_slow_time_variance(self):
        """
        Calculate slow-time variance for each range bin

        This helps identify the range bin with maximum chest motion,
        which corresponds to the location of the subject's chest

        Returns:
            1D array of variance values for each range bin
        """
        print("Calculating slow-time variance for each range bin...")

        
        
        magnitude = np.abs(self.processed_matrix)
        self.variance_profile = np.var(magnitude, axis=0)

        print(f"Variance profile shape: {self.variance_profile.shape}")

        return self.variance_profile

    def detect_chest_range_bin(self, method="variance", search_range=None):
        """
        Detect the range bin corresponding to the subject's chest

        Args:
            method: Detection method ('variance', 'max_amplitude')
            search_range: Tuple (min_bin, max_bin) to search within

        Returns:
            Index of the detected chest range bin
        """
        print(f"Detecting chest range bin using {method} method...")

        if method == "variance":
            if self.variance_profile is None:
                self.calculate_slow_time_variance()

            
            if search_range is not None:
                min_bin, max_bin = search_range
                search_variance = self.variance_profile[min_bin:max_bin]
                chest_bin = min_bin + np.argmax(search_variance)
            else:
                chest_bin = np.argmax(self.variance_profile)

        elif method == "max_amplitude":
            
            avg_amplitude = np.mean(np.abs(self.processed_matrix), axis=0)

            if search_range is not None:
                min_bin, max_bin = search_range
                search_amplitude = avg_amplitude[min_bin:max_bin]
                chest_bin = min_bin + np.argmax(search_amplitude)
            else:
                chest_bin = np.argmax(avg_amplitude)

        else:
            print(f"Unknown detection method: {method}")
            chest_bin = 0

        
        from range_time_matrix import RangeTimeMatrix
        dummy_rtm = RangeTimeMatrix(self.processed_matrix)
        dummy_rtm.construct_matrix()
        range_bins = dummy_rtm.get_range_bins()
        chest_range = range_bins[chest_bin]

        print(f"Detected chest at range bin {chest_bin} ({chest_range:.2f} m)")

        return chest_bin

    def normalize_range_bins(self):
        """
        Normalize each range bin independently

        This equalizes the amplitude across different range bins
        """
        print("Normalizing range bins...")

        normalized_matrix = np.zeros_like(self.processed_matrix)

        for range_bin in range(self.processed_matrix.shape[1]):
            bin_data = self.processed_matrix[:, range_bin]
            bin_magnitude = np.abs(bin_data)

            
            max_amplitude = np.max(bin_magnitude)

            if max_amplitude > 0:
                normalized_matrix[:, range_bin] = bin_data / max_amplitude
            else:
                normalized_matrix[:, range_bin] = bin_data

        self.processed_matrix = normalized_matrix

        print("Normalization complete")

        return self.processed_matrix

    def extract_range_bin(self, bin_index):
        """
        Extract time series from a specific range bin

        Args:
            bin_index: Index of the range bin to extract

        Returns:
            1D complex array representing the slow-time signal
        """
        return self.processed_matrix[:, bin_index]

    def get_processed_matrix(self):
        """Get the current processed matrix"""
        return self.processed_matrix

    def run_full_pipeline(self, clutter_method="mean_subtraction",
                         apply_highpass=True, normalize=True):
        """
        Run the complete preprocessing pipeline

        Args:
            clutter_method: Method for clutter removal
            apply_highpass: Whether to apply high-pass filtering
            normalize: Whether to normalize range bins

        Returns:
            Processed matrix and detected chest bin
        """
        print("\n" + "=" * 60)
        print("Running Full Preprocessing Pipeline")
        print("=" * 60)

        
        if clutter_method == "mean_subtraction":
            self.remove_clutter_mean_subtraction()
        elif clutter_method == "median_subtraction":
            self.remove_clutter_median_subtraction()
        elif clutter_method == "moving_average":
            self.remove_clutter_moving_average()
        else:
            print(f"Unknown clutter method: {clutter_method}")

        
        if apply_highpass:
            self.apply_highpass_filter()

        
        self.calculate_slow_time_variance()

        
        chest_bin = self.detect_chest_range_bin(method="variance")

        
        if normalize:
            self.normalize_range_bins()

        print("=" * 60)
        print("Preprocessing Pipeline Complete")
        print("=" * 60)

        return self.processed_matrix, chest_bin


def test_preprocessor():
    """Test function for preprocessor"""
    print("Testing Preprocessor Module")
    print("=" * 60)

    
    print("Generating synthetic data...")

    num_pulses = Config.NUM_PULSES
    samples_per_pulse = Config.SAMPLES_PER_PULSE

    
    target_range_bin = 200
    breathing_freq = 0.3  
    breathing_amplitude = 0.1  

    synthetic_data = np.zeros((num_pulses, samples_per_pulse), dtype=np.complex64)

    
    for i in range(samples_per_pulse):
        clutter_amplitude = np.random.rand() * 0.5
        synthetic_data[:, i] = clutter_amplitude

    
    time_axis = np.arange(num_pulses) / Config.PULSE_REPETITION_FREQ
    phase_modulation = breathing_amplitude * np.sin(2 * np.pi * breathing_freq * time_axis)

    for i in range(num_pulses):
        synthetic_data[i, target_range_bin] += 1.0 * np.exp(1j * phase_modulation[i])

    
    noise = 0.05 * (np.random.randn(num_pulses, samples_per_pulse) +
                    1j * np.random.randn(num_pulses, samples_per_pulse))
    synthetic_data += noise

    print(f"Synthetic data shape: {synthetic_data.shape}")

    
    preprocessor = Preprocessor(synthetic_data)
    processed_matrix, chest_bin = preprocessor.run_full_pipeline(
        clutter_method="mean_subtraction",
        apply_highpass=True,
        normalize=True
    )

    print(f"\nProcessed matrix shape: {processed_matrix.shape}")
    print(f"Detected chest bin: {chest_bin}")
    print(f"Expected chest bin: {target_range_bin}")

    
    chest_signal = preprocessor.extract_range_bin(chest_bin)
    print(f"Chest signal shape: {chest_signal.shape}")

    
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    np.save(f"{Config.OUTPUT_DIR}/preprocessed_matrix.npy", processed_matrix)
    np.save(f"{Config.OUTPUT_DIR}/chest_signal.npy", chest_signal)
    np.save(f"{Config.OUTPUT_DIR}/variance_profile.npy", preprocessor.variance_profile)

    print(f"\nSaved results to {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    Config.print_config()
    test_preprocessor()
