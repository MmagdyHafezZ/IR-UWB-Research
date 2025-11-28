#!/usr/bin/env python3
"""
Simplified Visualization Module
Process-safe matplotlib visualization without threading issues
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time


def run_visualization_standalone(data_queue):
    """
    Run visualization in standalone mode (for multiprocessing)

    Args:
        data_queue: Queue to receive data updates
    """
    # Set up matplotlib for non-blocking operation
    plt.ion()

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Real-Time Respiration Monitoring', fontsize=14, fontweight='bold')

    # Data storage
    breathing_rates = deque(maxlen=100)
    timestamps = deque(maxlen=100)
    snr_values = deque(maxlen=100)
    quality_scores = deque(maxlen=100)

    current_waveform = None
    variance_profile = None
    range_bins = None
    chest_bin = None

    # Initialize plot lines
    ax_rate = axes[0, 0]
    ax_wave = axes[0, 1]
    ax_snr = axes[1, 0]
    ax_quality = axes[1, 1]
    ax_variance = axes[2, 0]
    ax_hist = axes[2, 1]

    # Set up rate trend plot
    ax_rate.set_title('Breathing Rate Over Time')
    ax_rate.set_xlabel('Time (s)')
    ax_rate.set_ylabel('Rate (BPM)')
    ax_rate.grid(True, alpha=0.3)
    ax_rate.set_ylim(0, 30)
    line_rate, = ax_rate.plot([], [], 'b-', linewidth=2)

    # Set up waveform plot
    ax_wave.set_title('Current Breathing Waveform')
    ax_wave.set_xlabel('Time (s)')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.grid(True, alpha=0.3)
    line_wave, = ax_wave.plot([], [], 'g-', linewidth=1.5)

    # Set up SNR plot
    ax_snr.set_title('Signal-to-Noise Ratio')
    ax_snr.set_xlabel('Time (s)')
    ax_snr.set_ylabel('SNR (dB)')
    ax_snr.grid(True, alpha=0.3)
    line_snr, = ax_snr.plot([], [], 'm-', linewidth=2)
    ax_snr.axhline(y=-10, color='r', linestyle='--', alpha=0.5)

    # Set up quality plot
    ax_quality.set_title('Signal Quality Score')
    ax_quality.set_xlabel('Time (s)')
    ax_quality.set_ylabel('Quality (0-1)')
    ax_quality.grid(True, alpha=0.3)
    ax_quality.set_ylim(0, 1)
    line_quality, = ax_quality.plot([], [], 'orange', linewidth=2)
    ax_quality.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)

    # Set up variance plot
    ax_variance.set_title('Chest Detection (Variance Profile)')
    ax_variance.set_xlabel('Range (m)')
    ax_variance.set_ylabel('Variance')
    ax_variance.grid(True, alpha=0.3)
    line_variance, = ax_variance.plot([], [], 'b-', linewidth=1.5)
    line_chest = ax_variance.axvline(x=0, color='r', linestyle='--', linewidth=2)

    plt.tight_layout()

    print("[Visualization] Window opened")

    try:
        while plt.fignum_exists(fig.number):
            # Get new data from queue (non-blocking)
            try:
                while not data_queue.empty():
                    data = data_queue.get_nowait()

                    # Update data storage
                    breathing_rates.append(data.get('breathing_rate', 0))
                    timestamps.append(time.time())
                    snr_values.append(data.get('snr', -100))
                    quality_scores.append(data.get('quality_score', 0))
                    current_waveform = data.get('phase_waveform')
                    variance_profile = data.get('variance_profile')
                    range_bins = data.get('range_bins')
                    chest_bin = data.get('chest_bin')
            except:
                pass

            # Update plots
            if len(breathing_rates) > 0:
                # Rate trend
                t_rel = [(t - timestamps[0]) for t in timestamps]
                line_rate.set_data(t_rel, list(breathing_rates))
                if len(t_rel) > 0:
                    ax_rate.set_xlim(max(0, t_rel[-1] - 60), t_rel[-1] + 5)

                # SNR
                line_snr.set_data(t_rel, list(snr_values))
                if len(snr_values) > 0:
                    y_min = min(-20, min(snr_values) - 5)
                    y_max = max(10, max(snr_values) + 5)
                    ax_snr.set_ylim(y_min, y_max)
                    ax_snr.set_xlim(max(0, t_rel[-1] - 60), t_rel[-1] + 5)

                # Quality
                line_quality.set_data(t_rel, list(quality_scores))
                if len(t_rel) > 0:
                    ax_quality.set_xlim(max(0, t_rel[-1] - 60), t_rel[-1] + 5)

            # Waveform
            if current_waveform is not None:
                from config import Config
                n = len(current_waveform)
                t_wave = np.arange(n) / Config.PULSE_REPETITION_FREQ
                line_wave.set_data(t_wave, current_waveform)
                ax_wave.relim()
                ax_wave.autoscale_view()

            # Variance profile
            if variance_profile is not None and range_bins is not None:
                line_variance.set_data(range_bins, variance_profile)
                ax_variance.relim()
                ax_variance.autoscale_view()

                if chest_bin is not None and chest_bin < len(range_bins):
                    line_chest.set_xdata([range_bins[chest_bin], range_bins[chest_bin]])

            # Histogram
            if len(breathing_rates) >= 5:
                ax_hist.clear()
                rates = [r for r in breathing_rates if r > 0]
                if len(rates) >= 2:
                    ax_hist.hist(rates, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
                    ax_hist.axvline(np.mean(rates), color='red', linestyle='--',
                                   linewidth=2, label=f'Mean: {np.mean(rates):.1f} BPM')
                    ax_hist.set_xlabel('Rate (BPM)')
                    ax_hist.set_ylabel('Frequency')
                    ax_hist.set_title('Breathing Rate Distribution')
                    ax_hist.grid(True, alpha=0.3, axis='y')
                    ax_hist.legend()

            # Update display
            plt.pause(0.5)  # Update every 500ms

    except KeyboardInterrupt:
        print("[Visualization] Closed by user")
    except Exception as e:
        print(f"[Visualization] Error: {e}")
    finally:
        plt.close(fig)
        print("[Visualization] Window closed")


class SimpleDashboard:
    """Simple dashboard that doesn't use threading"""

    def __init__(self):
        self.process = None
        self.data_queue = None

    def start(self):
        """Start dashboard in separate process"""
        import multiprocessing as mp

        # Create queue for data passing
        self.data_queue = mp.Queue()

        # Start process
        self.process = mp.Process(
            target=run_visualization_standalone,
            args=(self.data_queue,),
            daemon=True
        )
        self.process.start()
        print("[Dashboard] Started in separate process")

    def update(self, data):
        """Send data to dashboard"""
        if self.data_queue:
            try:
                # Non-blocking put
                self.data_queue.put_nowait(data)
            except:
                pass  # Queue full, skip this update

    def stop(self):
        """Stop dashboard"""
        if self.process:
            self.process.terminate()
            self.process.join(timeout=2.0)
            self.process = None
            print("[Dashboard] Stopped")


def test_visualization():
    """Test visualization with synthetic data"""
    import multiprocessing as mp

    print("Testing visualization...")
    print("Close window or press Ctrl+C to exit")

    # Create queue
    data_queue = mp.Queue()

    # Start visualization process
    viz_process = mp.Process(
        target=run_visualization_standalone,
        args=(data_queue,),
        daemon=False
    )
    viz_process.start()

    # Send test data
    try:
        t0 = time.time()
        while viz_process.is_alive():
            t = time.time() - t0

            # Simulate breathing at 15 BPM
            breathing_rate = 15 + np.random.randn() * 2
            snr = -5 + np.random.randn() * 3
            quality_score = 0.1 + 0.05 * np.sin(2 * np.pi * 0.1 * t)

            # Simulate waveform
            waveform_time = np.linspace(0, 5, 500)
            waveform = np.sin(2 * np.pi * (breathing_rate/60) * waveform_time)

            # Simulate variance profile
            range_bins = np.linspace(0, 10, 512)
            variance_profile = np.random.rand(512) * 0.1
            chest_bin = 256
            variance_profile[chest_bin] = 1.0

            data = {
                'breathing_rate': breathing_rate,
                'snr': snr,
                'quality_score': quality_score,
                'phase_waveform': waveform,
                'variance_profile': variance_profile,
                'range_bins': range_bins,
                'chest_bin': chest_bin
            }

            data_queue.put(data)
            time.sleep(1)  # Send update every second

    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        viz_process.terminate()
        viz_process.join()
        print("Test complete")


if __name__ == "__main__":
    test_visualization()
