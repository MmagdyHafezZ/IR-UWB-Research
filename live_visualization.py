#!/usr/bin/env python3
"""
Live Visualization Module
Real-time plotting with matplotlib animation in separate thread
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import threading
from collections import deque
from queue import Queue, Empty
import time


class LivePlotter(threading.Thread):
    """Thread-safe live plotting with matplotlib"""

    def __init__(self, update_interval=1000):
        super().__init__(daemon=True)
        self.update_interval = update_interval  # milliseconds
        self.running = False

        # Data queues
        self.data_queue = Queue()

        # Data storage
        self.breathing_rates = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.snr_values = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)

        self.current_waveform = None
        self.variance_profile = None
        self.range_bins = None
        self.chest_bin = None

        # Plot elements
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.ani = None

    def run(self):
        """Start live plotting in separate thread"""
        self.running = True
        print("  Live plotter thread started")

        # Create figure
        self._setup_plots()

        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update_plots,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )

        plt.show()

    def _setup_plots(self):
        """Setup matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Real-Time Respiration Monitoring', fontsize=16, fontweight='bold')

        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        # Top left: Breathing rate trend
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.set_title('Breathing Rate Over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Rate (BPM)')
        ax1.grid(True, alpha=0.3)
        self.axes['rate_trend'] = ax1
        self.lines['rate_trend'], = ax1.plot([], [], 'b-', linewidth=2, label='Breathing Rate')
        self.lines['rate_avg'], = ax1.plot([], [], 'r--', linewidth=1.5, label='Average (10s)')
        ax1.legend()
        ax1.set_ylim(0, 30)

        # Top right: Current breathing waveform
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.set_title('Current Breathing Waveform')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        self.axes['waveform'] = ax2
        self.lines['waveform'], = ax2.plot([], [], 'g-', linewidth=1.5)

        # Middle left: SNR trend
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax3.set_title('Signal-to-Noise Ratio')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('SNR (dB)')
        ax3.grid(True, alpha=0.3)
        self.axes['snr'] = ax3
        self.lines['snr'], = ax3.plot([], [], 'm-', linewidth=2)
        ax3.axhline(y=-10, color='r', linestyle='--', alpha=0.5, label='Poor threshold')
        ax3.legend()

        # Middle right: Quality score
        ax4 = self.fig.add_subplot(gs[1, 1])
        ax4.set_title('Signal Quality Score')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Quality (0-1)')
        ax4.grid(True, alpha=0.3)
        self.axes['quality'] = ax4
        self.lines['quality'], = ax4.plot([], [], 'orange', linewidth=2)
        ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Minimum threshold')
        ax4.legend()
        ax4.set_ylim(0, 1)

        # Bottom left: Chest detection (variance profile)
        ax5 = self.fig.add_subplot(gs[2, 0])
        ax5.set_title('Chest Detection (Variance Profile)')
        ax5.set_xlabel('Range (m)')
        ax5.set_ylabel('Variance')
        ax5.grid(True, alpha=0.3)
        self.axes['variance'] = ax5
        self.lines['variance'], = ax5.plot([], [], 'b-', linewidth=1.5)
        self.lines['chest_marker'] = ax5.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Detected chest')
        ax5.legend()

        # Bottom right: Breathing rate histogram
        ax6 = self.fig.add_subplot(gs[2, 1])
        ax6.set_title('Breathing Rate Distribution')
        ax6.set_xlabel('Rate (BPM)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3, axis='y')
        self.axes['histogram'] = ax6

    def _update_plots(self, frame):
        """Animation update function"""
        # Process incoming data
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self._process_new_data(data)
            except Empty:
                break

        # Update plots
        self._update_rate_trend()
        self._update_waveform()
        self._update_snr()
        self._update_quality()
        self._update_variance()
        self._update_histogram()

        return []

    def _process_new_data(self, data):
        """Process new data from queue"""
        # Extract data
        timestamp = data.get('timestamp', time.time())
        breathing_rate = data.get('breathing_rate', 0)
        snr = data.get('snr', -100)
        quality_score = data.get('quality_score', 0)

        # Add to history
        self.timestamps.append(timestamp)
        self.breathing_rates.append(breathing_rate)
        self.snr_values.append(snr)
        self.quality_scores.append(quality_score)

        # Update current displays
        self.current_waveform = data.get('phase_waveform')
        self.variance_profile = data.get('variance_profile')
        self.range_bins = data.get('range_bins')
        self.chest_bin = data.get('chest_bin')

    def _update_rate_trend(self):
        """Update breathing rate trend plot"""
        if len(self.breathing_rates) == 0:
            return

        # Convert timestamps to relative seconds
        timestamps = list(self.timestamps)
        if len(timestamps) == 0:
            return

        t0 = timestamps[0]
        time_axis = [(t - t0).total_seconds() if hasattr(t, 'total_seconds') else (t - t0) for t in timestamps]
        rates = list(self.breathing_rates)

        # Update line
        self.lines['rate_trend'].set_data(time_axis, rates)

        # Calculate moving average
        if len(rates) >= 5:
            window = min(5, len(rates))
            avg_rates = []
            for i in range(len(rates)):
                start = max(0, i - window + 1)
                avg_rates.append(np.mean(rates[start:i+1]))
            self.lines['rate_avg'].set_data(time_axis, avg_rates)

        # Auto-scale x-axis
        ax = self.axes['rate_trend']
        if len(time_axis) > 0:
            ax.set_xlim(max(0, time_axis[-1] - 60), time_axis[-1] + 5)

    def _update_waveform(self):
        """Update breathing waveform plot"""
        if self.current_waveform is None:
            return

        # Create time axis (last 5 seconds)
        from config import Config
        sampling_rate = Config.PULSE_REPETITION_FREQ
        n_samples = len(self.current_waveform)
        time_axis = np.arange(n_samples) / sampling_rate

        self.lines['waveform'].set_data(time_axis, self.current_waveform)

        # Auto-scale
        ax = self.axes['waveform']
        ax.relim()
        ax.autoscale_view(True, True, True)

    def _update_snr(self):
        """Update SNR plot"""
        if len(self.snr_values) == 0:
            return

        timestamps = list(self.timestamps)
        if len(timestamps) == 0:
            return

        t0 = timestamps[0]
        time_axis = [(t - t0).total_seconds() if hasattr(t, 'total_seconds') else (t - t0) for t in timestamps]
        snr_vals = list(self.snr_values)

        self.lines['snr'].set_data(time_axis, snr_vals)

        # Auto-scale
        ax = self.axes['snr']
        if len(time_axis) > 0:
            ax.set_xlim(max(0, time_axis[-1] - 60), time_axis[-1] + 5)
        if len(snr_vals) > 0:
            y_min = min(-20, min(snr_vals) - 5)
            y_max = max(10, max(snr_vals) + 5)
            ax.set_ylim(y_min, y_max)

    def _update_quality(self):
        """Update quality score plot"""
        if len(self.quality_scores) == 0:
            return

        timestamps = list(self.timestamps)
        if len(timestamps) == 0:
            return

        t0 = timestamps[0]
        time_axis = [(t - t0).total_seconds() if hasattr(t, 'total_seconds') else (t - t0) for t in timestamps]
        quality_vals = list(self.quality_scores)

        self.lines['quality'].set_data(time_axis, quality_vals)

        # Auto-scale x
        ax = self.axes['quality']
        if len(time_axis) > 0:
            ax.set_xlim(max(0, time_axis[-1] - 60), time_axis[-1] + 5)

    def _update_variance(self):
        """Update variance profile plot"""
        if self.variance_profile is None or self.range_bins is None:
            return

        self.lines['variance'].set_data(self.range_bins, self.variance_profile)

        # Update chest marker
        if self.chest_bin is not None and self.chest_bin < len(self.range_bins):
            chest_range = self.range_bins[self.chest_bin]
            self.lines['chest_marker'].set_xdata([chest_range, chest_range])

        # Auto-scale
        ax = self.axes['variance']
        ax.relim()
        ax.autoscale_view(True, True, True)

    def _update_histogram(self):
        """Update breathing rate histogram"""
        if len(self.breathing_rates) < 5:
            return

        ax = self.axes['histogram']
        ax.clear()

        rates = [r for r in self.breathing_rates if r > 0]  # Filter out zeros
        if len(rates) < 2:
            return

        ax.hist(rates, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rates):.1f} BPM')
        ax.set_xlabel('Rate (BPM)')
        ax.set_ylabel('Frequency')
        ax.set_title('Breathing Rate Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    def update_data(self, data):
        """Add new data to queue (called from main thread)"""
        self.data_queue.put(data)

    def stop(self):
        """Stop plotting"""
        self.running = False
        if self.fig:
            plt.close(self.fig)


class LiveDashboard:
    """Comprehensive live dashboard with multiple visualization modes"""

    def __init__(self):
        self.plotter = None
        self.running = False

    def start(self):
        """Start dashboard"""
        if self.running:
            return

        self.plotter = LivePlotter(update_interval=1000)
        self.plotter.start()
        self.running = True

        print("  Live dashboard started")

    def update(self, data):
        """Update dashboard with new data"""
        if self.plotter and self.running:
            self.plotter.update_data(data)

    def stop(self):
        """Stop dashboard"""
        if self.plotter:
            self.plotter.stop()
        self.running = False


def test_live_plotting():
    """Test the live plotting functionality"""
    import time
    from datetime import datetime

    print("Testing live visualization...")
    print("Close the plot window to exit")

    dashboard = LiveDashboard()
    dashboard.start()

    # Simulate data stream
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0

            # Simulate breathing at 15 BPM with noise
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
                'timestamp': datetime.now(),
                'breathing_rate': breathing_rate,
                'snr': snr,
                'quality_score': quality_score,
                'phase_waveform': waveform,
                'variance_profile': variance_profile,
                'range_bins': range_bins,
                'chest_bin': chest_bin
            }

            dashboard.update(data)
            time.sleep(2)  # Update every 2 seconds

    except KeyboardInterrupt:
        print("\nStopping test...")
        dashboard.stop()


if __name__ == "__main__":
    test_live_plotting()
