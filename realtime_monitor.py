#!/usr/bin/env python3
"""
Real-Time Respiration Monitor - Interactive CLI Application
Fully threaded with live visualization and metrics
"""

import numpy as np
import time
import threading
import multiprocessing as mp
from queue import Queue, Empty
from collections import deque
from datetime import datetime
import os
import sys

from config import Config
from range_time_matrix import RangeTimeMatrix
from preprocessing import Preprocessor
from respiration_extraction import RespirationExtractor
from processing_fixes import (
    improved_chest_detection,
    improved_phase_extraction,
    improved_clutter_removal,
    diagnose_signal_quality
)
from live_visualization import LiveDashboard


class DataBuffer:
    """Thread-safe circular buffer for radar data"""

    def __init__(self, max_pulses=5000):
        self.max_pulses = max_pulses
        self.buffer = deque(maxlen=max_pulses)
        self.lock = threading.Lock()

    def add_pulse(self, pulse_data):
        """Add a single pulse to buffer"""
        with self.lock:
            self.buffer.append(pulse_data)

    def get_data_matrix(self):
        """Get current buffer as numpy array"""
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return np.array(list(self.buffer))

    def get_buffer_size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()


class MetricsTracker:
    """Track breathing metrics over time"""

    def __init__(self, history_length=60):
        self.history_length = history_length
        self.breathing_rates = deque(maxlen=history_length)
        self.snr_values = deque(maxlen=history_length)
        self.quality_scores = deque(maxlen=history_length)
        self.timestamps = deque(maxlen=history_length)
        self.lock = threading.Lock()

    def add_measurement(self, breathing_rate, snr, quality_score, timestamp=None):
        """Add new measurement with optional timestamp"""
        with self.lock:
            self.breathing_rates.append(breathing_rate)
            self.snr_values.append(snr)
            self.quality_scores.append(quality_score)
            # Use provided timestamp or generate new one
            if timestamp is None:
                timestamp = time.time()
            self.timestamps.append(timestamp)

    def get_current_rate(self):
        """Get most recent breathing rate"""
        with self.lock:
            if len(self.breathing_rates) == 0:
                return 0.0
            return self.breathing_rates[-1]

    def get_average_rate(self, window=10):
        """Get average rate over last N measurements"""
        with self.lock:
            if len(self.breathing_rates) == 0:
                return 0.0
            recent = list(self.breathing_rates)[-window:]
            return np.mean(recent)

    def get_rate_variability(self):
        """Get standard deviation of breathing rate"""
        with self.lock:
            if len(self.breathing_rates) < 2:
                return 0.0
            return np.std(list(self.breathing_rates))

    def get_trend_data(self):
        """Get data for trend plotting"""
        with self.lock:
            return {
                'timestamps': list(self.timestamps),
                'breathing_rates': list(self.breathing_rates),
                'snr_values': list(self.snr_values),
                'quality_scores': list(self.quality_scores)
            }


class ProcessingWorkerPool:
    """Multiprocessing pool for heavy computations"""

    def __init__(self, num_workers=None):
        if num_workers is None:
            num_workers = max(2, mp.cpu_count() - 1)
        self.pool = mp.Pool(processes=num_workers)
        print(f"  Initialized processing pool with {num_workers} workers")

    def process_vmd_async(self, signal_data, callback):
        """Process VMD decomposition asynchronously"""
        self.pool.apply_async(
            process_vmd_worker,
            args=(signal_data,),
            callback=callback
        )

    def shutdown(self):
        """Shutdown pool"""
        self.pool.close()
        self.pool.join()


def process_vmd_worker(signal_data):
    """Worker function for VMD processing (runs in separate process)"""
    from vmd import vmd

    u, u_hat, omega = vmd(
        signal_data,
        alpha=Config.VMD_ALPHA,
        tau=Config.VMD_TAU,
        K=Config.VMD_NUM_MODES,
        DC=Config.VMD_DC_PART,
        init=Config.VMD_INIT_METHOD,
        tol=Config.VMD_TOL,
        max_iter=Config.VMD_MAX_ITER
    )

    return {'modes': u, 'modes_hat': u_hat, 'center_freqs': omega}


class CaptureThread(threading.Thread):
    """Thread for continuous data capture"""

    def __init__(self, data_buffer, use_hardware=False):
        super().__init__(daemon=True)
        self.data_buffer = data_buffer
        self.use_hardware = use_hardware
        self.running = False
        self.paused = False
        self.pulse_count = 0

    def run(self):
        """Main capture loop"""
        self.running = True
        print(f"  Capture thread started ({'hardware' if self.use_hardware else 'synthetic'} mode)")

        if self.use_hardware:
            self._capture_from_hardware()
        else:
            self._capture_synthetic()

    def _capture_from_hardware(self):
        """Capture from SDR hardware"""
        try:
            from sdr_capture import SDRCapture
            sdr = SDRCapture(hardware_mode=True)
            sdr.setup_transmitter()
            sdr.setup_receiver()
            sdr.generate_impulse_signal()

            while self.running:
                if not self.paused:
                    # Capture single pulse
                    pulse_data = sdr.record_single_pulse()
                    self.data_buffer.add_pulse(pulse_data)
                    self.pulse_count += 1
                time.sleep(1.0 / Config.PULSE_REPETITION_FREQ)

        except Exception as e:
            print(f"  Hardware capture error: {e}")
            print("  Falling back to synthetic mode")
            self._capture_synthetic()

    def _capture_synthetic(self):
        """Generate synthetic data stream"""
        # Simulation parameters
        samples_per_pulse = Config.SAMPLES_PER_PULSE
        chest_bin = 256
        breathing_rate_bpm = 15 + np.random.randn() * 2  # Variable rate

        # Static clutter
        np.random.seed(int(time.time()))
        clutter = np.zeros(samples_per_pulse, dtype=np.complex64)
        for i in range(samples_per_pulse):
            clutter_amplitude = 0.3
            clutter_phase = np.random.rand() * 2 * np.pi
            clutter[i] = clutter_amplitude * np.exp(1j * clutter_phase)

        start_time = time.time()

        while self.running:
            if not self.paused:
                current_time = time.time() - start_time

                # Breathing signal
                breathing_freq = breathing_rate_bpm / 60
                displacement_mm = 6.0 * np.sin(2 * np.pi * breathing_freq * current_time)

                # Add slow drift
                drift = 0.3 * np.sin(2 * np.pi * 0.05 * current_time)

                # Convert to phase
                wavelength_mm = 122.0
                phase_mod = (4 * np.pi / wavelength_mm) * (displacement_mm + drift)

                # Create pulse
                pulse = clutter.copy()
                chest_amplitude = 5.0 + np.random.randn() * 0.5
                pulse[chest_bin] = chest_amplitude * np.exp(1j * phase_mod)

                # Add noise
                pulse += 0.05 * (np.random.randn(samples_per_pulse) +
                                1j * np.random.randn(samples_per_pulse))

                self.data_buffer.add_pulse(pulse)
                self.pulse_count += 1

            time.sleep(1.0 / Config.PULSE_REPETITION_FREQ)

    def pause(self):
        """Pause capture"""
        self.paused = True

    def resume(self):
        """Resume capture"""
        self.paused = False

    def stop(self):
        """Stop capture"""
        self.running = False


class ProcessingThread(threading.Thread):
    """Thread for continuous signal processing"""

    def __init__(self, data_buffer, metrics_tracker, results_queue):
        super().__init__(daemon=True)
        self.data_buffer = data_buffer
        self.metrics_tracker = metrics_tracker
        self.results_queue = results_queue
        self.running = False
        self.processing_interval = 2.0  # Process every 2 seconds
        self.min_pulses = 1000  # Minimum pulses needed

    def run(self):
        """Main processing loop"""
        self.running = True
        print(f"  Processing thread started (interval: {self.processing_interval}s)")

        while self.running:
            time.sleep(self.processing_interval)

            # Check if enough data
            buffer_size = self.data_buffer.get_buffer_size()
            if buffer_size < self.min_pulses:
                continue

            # Process data
            try:
                self._process_current_buffer()
            except Exception as e:
                print(f"  Processing error: {e}")
                import traceback
                traceback.print_exc()

    def _process_current_buffer(self):
        """Process current buffer data"""
        # Get data
        raw_data = self.data_buffer.get_data_matrix()
        if raw_data is None:
            return

        # Construct RTM
        rtm = RangeTimeMatrix(raw_data)
        rtm_matrix = rtm.construct_matrix()

        # Improved clutter removal
        clutter_removed, _ = improved_clutter_removal(
            rtm_matrix,
            method='moving_average',
            window_size=50
        )

        # Improved chest detection
        preprocessor = Preprocessor(clutter_removed)
        preprocessor.calculate_slow_time_variance()

        range_bins = rtm.get_range_bins()
        chest_bin, chest_info = improved_chest_detection(
            preprocessor.variance_profile,
            range_bins,
            smoothing_sigma=5.0,
            prominence_factor=2.0,
            min_range_m=0.3,
            max_range_m=3.0
        )

        # Extract chest signal
        chest_signal = clutter_removed[:, chest_bin]

        # Improved phase extraction
        phase_cleaned, phase_info = improved_phase_extraction(
            chest_signal,
            Config.PULSE_REPETITION_FREQ,
            detrend=True,
            remove_dc=True,
            normalize=False
        )

        # Bandpass filter
        from scipy import signal
        nyquist = Config.PULSE_REPETITION_FREQ / 2
        low = Config.BREATHING_FREQ_MIN / nyquist
        high = Config.BREATHING_FREQ_MAX / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        phase_filtered = signal.filtfilt(b, a, phase_cleaned)

        # Extract breathing rate
        extractor = RespirationExtractor(phase_filtered, Config.PULSE_REPETITION_FREQ)
        rate_time = extractor.detect_breathing_rate_time_domain()
        rate_freq = extractor.detect_breathing_rate_frequency_domain()

        # Use frequency domain if available, fallback to time domain
        breathing_rate = rate_freq if rate_freq > 0 else rate_time

        # Calculate quality metrics
        quality_report = diagnose_signal_quality(chest_signal, Config.PULSE_REPETITION_FREQ)
        snr = 10 * np.log10(quality_report['spectral']['breathing_fraction'] + 1e-10)
        quality_score = quality_report['spectral']['breathing_fraction']

        # Create timestamp for this measurement
        measurement_time = datetime.now()
        measurement_timestamp = measurement_time.timestamp()

        # Update metrics with consistent timestamp
        self.metrics_tracker.add_measurement(breathing_rate, snr, quality_score, measurement_timestamp)

        # Send results to display
        results = {
            'breathing_rate': breathing_rate,
            'snr': snr,
            'quality_score': quality_score,
            'chest_bin': chest_bin,
            'chest_range': range_bins[chest_bin],
            'phase_waveform': phase_filtered[-500:],  # Last 500 samples
            'variance_profile': chest_info['smoothed_variance'],
            'range_bins': range_bins,
            'buffer_size': len(raw_data),
            'timestamp': measurement_time
        }

        self.results_queue.put(results)

    def stop(self):
        """Stop processing"""
        self.running = False


class RealTimeMonitor:
    """Main real-time monitoring application"""

    def __init__(self, use_hardware=False):
        self.use_hardware = use_hardware
        self.data_buffer = DataBuffer(max_pulses=5000)
        self.metrics_tracker = MetricsTracker(history_length=60)
        self.results_queue = Queue()

        self.capture_thread = None
        self.processing_thread = None
        self.dashboard = None

        self.running = False
        self.visualization_enabled = False

    def start(self):
        """Start monitoring"""
        print("\n" + "="*70)
        print("STARTING REAL-TIME MONITOR")
        print("="*70)

        self.running = True

        # Start capture thread
        self.capture_thread = CaptureThread(self.data_buffer, self.use_hardware)
        self.capture_thread.start()

        # Start processing thread
        self.processing_thread = ProcessingThread(
            self.data_buffer,
            self.metrics_tracker,
            self.results_queue
        )
        self.processing_thread.start()

        print("\n✓ Monitoring started")
        print(f"  Capture mode: {'Hardware' if self.use_hardware else 'Synthetic'}")
        print(f"  Buffer size: {self.data_buffer.max_pulses} pulses")
        print(f"  Processing interval: {self.processing_thread.processing_interval}s")

    def stop(self):
        """Stop monitoring"""
        print("\n[Stopping monitoring...]")

        self.running = False

        if self.capture_thread:
            self.capture_thread.stop()

        if self.processing_thread:
            self.processing_thread.stop()

        print("✓ Monitoring stopped")

    def get_latest_results(self, timeout=0.1):
        """Get latest processing results"""
        try:
            return self.results_queue.get(timeout=timeout)
        except Empty:
            return None

    def toggle_visualization(self):
        """Toggle live visualization on/off"""
        if self.visualization_enabled:
            # Stop visualization
            if self.dashboard:
                self.dashboard.stop()
                self.dashboard = None
            self.visualization_enabled = False
            print("✓ Visualization stopped")
        else:
            # Start visualization
            self.dashboard = LiveDashboard()
            self.dashboard.start()
            self.visualization_enabled = True
            print("✓ Visualization started (close plot window to stop)")

    def pause_capture(self):
        """Pause data capture"""
        if self.capture_thread:
            self.capture_thread.pause()
            print("✓ Capture paused")

    def resume_capture(self):
        """Resume data capture"""
        if self.capture_thread:
            self.capture_thread.resume()
            print("✓ Capture resumed")

    def reset_buffer(self):
        """Reset data buffer"""
        self.data_buffer.clear()
        print("✓ Buffer cleared")

    def export_data(self, filename=None):
        """Export current data and metrics"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Config.OUTPUT_DIR}/export_{timestamp}.npz"

        # Get data
        data_matrix = self.data_buffer.get_data_matrix()
        trend_data = self.metrics_tracker.get_trend_data()

        if data_matrix is not None:
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            np.savez(
                filename,
                raw_data=data_matrix,
                breathing_rates=trend_data['breathing_rates'],
                snr_values=trend_data['snr_values'],
                quality_scores=trend_data['quality_scores'],
                timestamps=trend_data['timestamps']
            )
            print(f"✓ Data exported to: {filename}")
            return filename
        else:
            print("✗ No data to export")
            return None

    def get_status(self):
        """Get current status"""
        return {
            'running': self.running,
            'buffer_size': self.data_buffer.get_buffer_size(),
            'pulse_count': self.capture_thread.pulse_count if self.capture_thread else 0,
            'current_rate': self.metrics_tracker.get_current_rate(),
            'average_rate': self.metrics_tracker.get_average_rate(),
            'rate_variability': self.metrics_tracker.get_rate_variability(),
            'visualization': self.visualization_enabled
        }


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_header():
    """Print application header"""
    clear_screen()
    print("="*70)
    print(" "*15 + "REAL-TIME RESPIRATION MONITOR")
    print("="*70)
    print()


def print_live_metrics(monitor, latest_results=None):
    """Print live metrics display"""
    status = monitor.get_status()
    trend_data = monitor.metrics_tracker.get_trend_data()

    print(f"Status: {'🟢 RUNNING' if status['running'] else '🔴 STOPPED'}")
    print(f"Visualization: {'🟢 ON' if status['visualization'] else '⚪ OFF'}")
    print(f"Buffer: {status['buffer_size']:5d} pulses | Total captured: {status['pulse_count']:6d}")
    print()

    print("─" * 70)
    print("CURRENT BREATHING METRICS")
    print("─" * 70)

    current_rate = status['current_rate']
    avg_rate = status['average_rate']
    variability = status['rate_variability']

    print(f"  Current Rate:    {current_rate:6.1f} BPM")
    print(f"  Average Rate:    {avg_rate:6.1f} BPM  (last 10 measurements)")
    print(f"  Variability:     {variability:6.1f} BPM  (std dev)")
    print()

    # Breathing rate bar
    if current_rate > 0:
        bar_length = int(current_rate / 30 * 50)  # Scale 0-30 BPM to 0-50 chars
        bar = "█" * min(bar_length, 50)
        print(f"  Rate Visual: [{bar:<50}] {current_rate:.1f} BPM")
    print()

    # Latest result details
    if latest_results:
        print("─" * 70)
        print("LATEST MEASUREMENT DETAILS")
        print("─" * 70)
        print(f"  Chest Range:     {latest_results['chest_range']:.2f} m (bin {latest_results['chest_bin']})")
        print(f"  SNR:             {latest_results['snr']:.1f} dB")
        print(f"  Quality Score:   {latest_results['quality_score']:.3f}")
        print(f"  Timestamp:       {latest_results['timestamp'].strftime('%H:%M:%S')}")
        print()

    # History
    if len(trend_data['breathing_rates']) > 0:
        print("─" * 70)
        print("RECENT MEASUREMENTS (last 10)")
        print("─" * 70)
        recent = list(trend_data['breathing_rates'])[-10:]
        for i, rate in enumerate(reversed(recent)):
            ago = (len(recent) - i - 1) * 2
            print(f"  {ago:3d}s ago: {rate:6.1f} BPM")

    print()
    print("="*70)


def interactive_menu():
    """Display interactive menu"""
    print("\nCOMMANDS:")
    print("  [s] Start/Resume monitoring")
    print("  [p] Pause monitoring")
    print("  [r] Reset buffer")
    print("  [m] Show metrics")
    print("  [e] Export data")
    print("  [v] Toggle visualization")
    print("  [q] Quit")
    print()
    return input("Enter command: ").strip().lower()


def main():
    """Main interactive CLI"""
    print_header()

    print("IR-UWB Real-Time Respiration Monitor")
    print("Interactive CLI Application with Threading & Multiprocessing")
    print()

    # Check hardware availability
    try:
        import SoapySDR
        print("✓ SoapySDR detected")
        use_hw = input("Use hardware? (y/n, default=n): ").strip().lower() == 'y'
    except ImportError:
        print("⚠ SoapySDR not installed - using synthetic mode")
        use_hw = False

    # Create monitor
    monitor = RealTimeMonitor(use_hardware=use_hw)

    # Auto-start
    monitor.start()

    # Variable to store latest results
    latest_results = None

    # Auto-refresh mode
    auto_refresh = True
    last_refresh = time.time()
    refresh_interval = 2.0

    print("\nMonitor started! Commands: [p]ause [r]esume [v]isualize [e]xport [x]clear [h]elp [q]uit")

    try:
        # Main loop
        while True:
            # Check for new results
            results = monitor.get_latest_results(timeout=0.1)
            if results:
                latest_results = results
                # Update dashboard if enabled
                if monitor.visualization_enabled and monitor.dashboard:
                    monitor.dashboard.update(results)

            # Auto-refresh display
            if auto_refresh and (time.time() - last_refresh) >= refresh_interval:
                print_header()
                print_live_metrics(monitor, latest_results)
                print("\nCommands: [p]ause [r]esume [v]isualize [e]xport [x]clear [h]elp [q]uit")
                print(">>> ", end='', flush=True)
                last_refresh = time.time()

            # Check for user input (non-blocking)
            import select
            if sys.platform != 'win32':
                # Unix-like systems
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    command = sys.stdin.readline().strip().lower()
                    handle_command(monitor, command, latest_results)
                    last_refresh = 0  # Force refresh after command
            else:
                # Windows - blocking input with timeout
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        command = input().strip().lower()
                        handle_command(monitor, command, latest_results)
                        last_refresh = 0
                except ImportError:
                    # Fallback: simple blocking input
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected]")

    finally:
        monitor.stop()
        print("\nThank you for using Real-Time Respiration Monitor!")
        print("="*70)


def handle_command(monitor, command, latest_results):
    """Handle user commands"""
    if command == 'q' or command == 'quit':
        print("\n[Quitting...]")
        sys.exit(0)

    elif command == 'p' or command == 'pause':
        monitor.pause_capture()

    elif command == 'r' or command == 'resume':
        monitor.resume_capture()

    elif command == 'v' or command == 'visualize':
        monitor.toggle_visualization()

    elif command == 'e' or command == 'export':
        monitor.export_data()

    elif command == 'x' or command == 'clear':
        monitor.reset_buffer()

    elif command == 's' or command == 'status':
        print("\n" + "="*70)
        print("SYSTEM STATUS")
        print("="*70)
        status = monitor.get_status()
        for key, value in status.items():
            print(f"  {key:20s}: {value}")
        print("="*70)

    elif command == 'h' or command == 'help':
        print("\n" + "="*70)
        print("AVAILABLE COMMANDS")
        print("="*70)
        print("  p, pause      - Pause data capture")
        print("  r, resume     - Resume data capture")
        print("  v, visualize  - Toggle live visualization (plots)")
        print("  e, export     - Export current data to file")
        print("  x, clear      - Clear data buffer")
        print("  s, status     - Show system status")
        print("  h, help       - Show this help message")
        print("  q, quit       - Exit application")
        print("="*70)

    elif command == '':
        # Just refresh
        pass

    else:
        print(f"Unknown command: '{command}'. Type 'h' for help.")


if __name__ == "__main__":
    main()
