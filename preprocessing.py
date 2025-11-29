"""
Preprocessing Pipeline Module
Handles clutter removal, filtering, and signal conditioning
"""

from scipy.ndimage import uniform_filter1d
from config import Config
import numpy as np
from scipy import signal


def _fallback_find_peaks(x, height=None, distance=None, prominence=None):
    """
    Robust local-maximum peak finder as a fallback for old SciPy.
    Supports height, distance, and basic prominence filtering.
    """
    x = np.asarray(x).ravel()
    if len(x) < 3:
        return np.array([], dtype=int), {}

    # Basic local maxima detection
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1

    # Height filtering
    if height is not None:
        if isinstance(height, dict):
            hmin = height.get("min", None)
        else:
            hmin = height
        if hmin is not None:
            peaks = peaks[x[peaks] >= hmin]

    # Distance filtering - keep only peaks separated by at least 'distance'
    if distance is not None and len(peaks) > 1:
        distance = int(distance)
        sorted_idx = np.argsort(x[peaks])[::-1]
        keep = np.ones(len(peaks), dtype=bool)
        for i, idx in enumerate(sorted_idx):
            if not keep[idx]:
                continue
            peak_pos = peaks[idx]
            for j in range(i + 1, len(sorted_idx)):
                other_idx = sorted_idx[j]
                if keep[other_idx] and abs(peaks[other_idx] - peak_pos) < distance:
                    keep[other_idx] = False
        peaks = peaks[keep]
        peaks = np.sort(peaks)

    # Prominence filtering (simplified)
    if prominence is not None and len(peaks) > 0:
        prominences = []
        for peak in peaks:
            left_min = np.min(x[max(0, peak-10):peak]) if peak > 0 else x[peak]
            right_min = np.min(x[peak+1:min(len(x), peak+11)]) if peak < len(x)-1 else x[peak]
            prom = x[peak] - max(left_min, right_min)
            prominences.append(prom)
        prominences = np.array(prominences)
        peaks = peaks[prominences >= prominence]

    properties = {}
    if height is not None and len(peaks) > 0:
        properties["peak_heights"] = x[peaks]

    return peaks, properties


if not hasattr(signal, "find_peaks"):
    signal.find_peaks = _fallback_find_peaks


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
        """Remove static clutter using per-bin mean subtraction"""
        print("Removing clutter using per-bin mean subtraction...")

        mean_profile = np.mean(self.processed_matrix, axis=0, keepdims=True)
        self.clutter_removed_matrix = self.processed_matrix - mean_profile

        print(f"Clutter removal complete. Matrix shape: {self.clutter_removed_matrix.shape}")

        self.processed_matrix = self.clutter_removed_matrix

        return self.clutter_removed_matrix

    def remove_clutter_median_subtraction(self):
        """Remove static clutter using per-bin median subtraction"""
        print("Removing clutter using per-bin median subtraction...")

        median_profile = np.median(self.processed_matrix, axis=0, keepdims=True)
        self.clutter_removed_matrix = self.processed_matrix - median_profile

        print(f"Clutter removal complete. Matrix shape: {self.clutter_removed_matrix.shape}")

        self.processed_matrix = self.clutter_removed_matrix

        return self.clutter_removed_matrix

    def remove_clutter_moving_average(self, window_size=50):
        """Remove clutter using moving average filter"""
        print(f"Removing clutter using moving average (window={window_size})...")

        clutter_estimate = uniform_filter1d(self.processed_matrix, size=window_size, axis=0, mode='nearest')
        self.clutter_removed_matrix = self.processed_matrix - clutter_estimate

        print(f"Clutter removal complete. Matrix shape: {self.clutter_removed_matrix.shape}")

        self.processed_matrix = self.clutter_removed_matrix

        return self.clutter_removed_matrix

    def apply_highpass_filter(self, cutoff_freq=None, order=4):
        """Apply slow-time high-pass filter for detrending"""
        if cutoff_freq is None:
            cutoff_freq = Config.HIGHPASS_CUTOFF

        print(f"Applying high-pass filter (cutoff={cutoff_freq} Hz, order={order})...")

        fs_slow = Config.PULSE_REPETITION_FREQ
        nyquist = fs_slow / 2
        normalized_cutoff = cutoff_freq / nyquist

        if normalized_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency {cutoff_freq} Hz is above Nyquist {nyquist} Hz")
            return self.processed_matrix

        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)

        filtered_matrix = np.zeros_like(self.processed_matrix)

        for range_bin in range(self.processed_matrix.shape[1]):
            if np.iscomplexobj(self.processed_matrix):
                real_filtered = signal.filtfilt(b, a, self.processed_matrix[:, range_bin].real)
                imag_filtered = signal.filtfilt(b, a, self.processed_matrix[:, range_bin].imag)
                filtered_matrix[:, range_bin] = real_filtered + 1j * imag_filtered
            else:
                filtered_matrix[:, range_bin] = signal.filtfilt(b, a, self.processed_matrix[:, range_bin])

        self.filtered_matrix = filtered_matrix
        self.processed_matrix = filtered_matrix

        print(f"High-pass filtering complete. Matrix shape: {self.processed_matrix.shape}")

        return self.filtered_matrix

    def calculate_slow_time_variance(self):
        """Calculate slow-time variance for each range bin"""
        print("Calculating slow-time variance for each range bin...")

        magnitude = np.abs(self.processed_matrix)
        self.variance_profile = np.var(magnitude, axis=0)

        print(f"Variance profile shape: {self.variance_profile.shape}")

        return self.variance_profile

    def detect_chest_range_bin(self, method="variance", search_range=None,
                               min_range_m=0.3, max_range_m=3.0):
        """Detect the range bin corresponding to the subject's chest"""
        print(f"Detecting chest range bin using {method} method...")

        if method == "variance":
            if self.variance_profile is None:
                self.calculate_slow_time_variance()

            if search_range is None:
                from range_time_matrix import RangeTimeMatrix
                dummy_rtm = RangeTimeMatrix(self.processed_matrix)
                dummy_rtm.construct_matrix()
                range_bins = dummy_rtm.get_range_bins()
                min_bin = int(np.searchsorted(range_bins, min_range_m))
                max_bin = int(np.searchsorted(range_bins, max_range_m))
                search_range = (min_bin, max_bin if max_bin > min_bin else len(range_bins))

            min_bin, max_bin = search_range
            min_bin = max(0, min_bin)
            max_bin = min(len(self.variance_profile), max_bin)
            if min_bin >= max_bin:
                min_bin, max_bin = 0, len(self.variance_profile)

            search_variance = self.variance_profile[min_bin:max_bin]
            chest_bin = min_bin + np.argmax(search_variance)

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
        chest_range = range_bins[chest_bin] if chest_bin < len(range_bins) else 0

        print(f"Detected chest at range bin {chest_bin} ({chest_range:.2f} m)")

        return chest_bin

    def normalize_range_bins(self):
        """Normalize each range bin independently"""
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
        """Extract time series from a specific range bin"""
        return self.processed_matrix[:, bin_index]

    def get_processed_matrix(self):
        """Get the current processed matrix"""
        return self.processed_matrix

    def run_full_pipeline(self, clutter_method="mean_subtraction",
                         apply_highpass=True, normalize=True):
        """Run the complete preprocessing pipeline"""
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

        chest_bin = self.detect_chest_range_bin(method="variance",
                                                min_range_m=0.5,
                                                max_range_m=3.0)

        if normalize:
            self.normalize_range_bins()

        print("=" * 60)
        print("Preprocessing Pipeline Complete")
        print("=" * 60)

        return self.processed_matrix, chest_bin


if __name__ == "__main__":
    Config.print_config()
    print("Preprocessing module loaded.")