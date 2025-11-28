#!/usr/bin/env python3
"""
Real-Time Respiration Monitor - Improved CLI using Rich library
Fixes terminal refresh/overwrite issues and provides proper TUI
"""

import numpy as np
import time
import threading
from queue import Queue, Empty
from collections import deque
from datetime import datetime
import os
import sys

# Rich library imports
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.prompt import Prompt, Confirm

# Application imports
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

# Global console
console = Console()


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
            buffer_copy = list(self.buffer)
        return np.array(buffer_copy, dtype=np.complex64)

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


class CaptureThread(threading.Thread):
    """Thread for continuous data capture"""

    def __init__(self, data_buffer, use_hardware=False):
        super().__init__(daemon=True)
        self.data_buffer = data_buffer
        self.use_hardware = use_hardware
        self.running = False
        self.paused = False
        self.pulse_count = 0
        self.status_message = "Initializing..."

    def run(self):
        """Main capture loop"""
        self.running = True
        self.status_message = f"Running ({'hardware' if self.use_hardware else 'synthetic'} mode)"

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
                    pulse_data = sdr.record_single_pulse()
                    self.data_buffer.add_pulse(pulse_data)
                    self.pulse_count += 1
                time.sleep(1.0 / Config.PULSE_REPETITION_FREQ)

        except Exception as e:
            self.status_message = f"Hardware error: {e}, falling back to synthetic"
            self._capture_synthetic()

    def _capture_synthetic(self):
        """Generate synthetic data stream"""
        samples_per_pulse = Config.SAMPLES_PER_PULSE
        chest_bin = 256
        breathing_rate_bpm = 15 + np.random.randn() * 2

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
                breathing_freq = breathing_rate_bpm / 60
                displacement_mm = 6.0 * np.sin(2 * np.pi * breathing_freq * current_time)
                drift = 0.3 * np.sin(2 * np.pi * 0.05 * current_time)
                wavelength_mm = 122.0
                phase_mod = (4 * np.pi / wavelength_mm) * (displacement_mm + drift)

                pulse = clutter.copy()
                chest_amplitude = 5.0 + np.random.randn() * 0.5
                pulse[chest_bin] = chest_amplitude * np.exp(1j * phase_mod)
                pulse += 0.05 * (np.random.randn(samples_per_pulse) +
                                1j * np.random.randn(samples_per_pulse))

                self.data_buffer.add_pulse(pulse)
                self.pulse_count += 1

            time.sleep(1.0 / Config.PULSE_REPETITION_FREQ)

    def pause(self):
        """Pause capture"""
        self.paused = True
        self.status_message = "Paused"

    def resume(self):
        """Resume capture"""
        self.paused = False
        self.status_message = f"Running ({'hardware' if self.use_hardware else 'synthetic'} mode)"

    def stop(self):
        """Stop capture"""
        self.running = False
        self.status_message = "Stopped"


class ProcessingThread(threading.Thread):
    """Thread for continuous signal processing"""

    def __init__(self, data_buffer, metrics_tracker, results_queue):
        super().__init__(daemon=True)
        self.data_buffer = data_buffer
        self.metrics_tracker = metrics_tracker
        self.results_queue = results_queue
        self.running = False
        self.processing_interval = 2.0
        self.min_pulses = 1000
        self.status_message = "Waiting for data..."
        self.last_error = None

    def run(self):
        """Main processing loop"""
        self.running = True
        self.status_message = "Running"

        while self.running:
            time.sleep(self.processing_interval)

            buffer_size = self.data_buffer.get_buffer_size()
            if buffer_size < self.min_pulses:
                self.status_message = f"Buffering... ({buffer_size}/{self.min_pulses})"
                continue

            try:
                self.status_message = "Processing..."
                self._process_current_buffer()
                self.status_message = "Ready"
                self.last_error = None
            except Exception as e:
                self.last_error = str(e)
                self.status_message = f"Error: {str(e)[:50]}"

    def _process_current_buffer(self):
        """Process current buffer data"""
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

        try:
            b, a = signal.butter(4, [low, high], btype='band')
            phase_filtered = signal.filtfilt(b, a, phase_cleaned)
        except Exception as e:
            # Fallback to unfiltered
            phase_filtered = phase_cleaned

        # Extract breathing rate
        extractor = RespirationExtractor(phase_filtered, Config.PULSE_REPETITION_FREQ)
        rate_time = extractor.detect_breathing_rate_time_domain()
        rate_freq = extractor.detect_breathing_rate_frequency_domain()

        breathing_rate = rate_freq if rate_freq > 0 else rate_time

        # Calculate quality metrics
        quality_report = diagnose_signal_quality(chest_signal, Config.PULSE_REPETITION_FREQ)
        snr = 10 * np.log10(quality_report['spectral']['breathing_fraction'] + 1e-10)
        quality_score = quality_report['spectral']['breathing_fraction']

        # Create timestamp
        measurement_time = datetime.now()
        measurement_timestamp = measurement_time.timestamp()

        # Update metrics
        self.metrics_tracker.add_measurement(breathing_rate, snr, quality_score, measurement_timestamp)

        # Send results
        results = {
            'breathing_rate': breathing_rate,
            'snr': snr,
            'quality_score': quality_score,
            'chest_bin': chest_bin,
            'chest_range': range_bins[chest_bin],
            'phase_waveform': phase_filtered[-500:],
            'variance_profile': chest_info['smoothed_variance'],
            'range_bins': range_bins,
            'buffer_size': len(raw_data),
            'timestamp': measurement_time
        }

        self.results_queue.put(results)

    def stop(self):
        """Stop processing"""
        self.running = False
        self.status_message = "Stopped"


class RichCLIMonitor:
    """Improved CLI monitor using Rich library"""

    def __init__(self, use_hardware=False):
        self.use_hardware = use_hardware
        self.data_buffer = DataBuffer(max_pulses=5000)
        self.metrics_tracker = MetricsTracker(history_length=60)
        self.results_queue = Queue()

        self.capture_thread = None
        self.processing_thread = None
        self.visualization_process = None
        self.viz_queue = None  # Queue for sending data to visualization

        self.running = False
        self.latest_results = None

    def start(self):
        """Start monitoring"""
        console.print("\n[bold green]Starting Real-Time Monitor[/bold green]")
        console.print(f"Mode: {'Hardware' if self.use_hardware else 'Synthetic'}\n")

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

        console.print("[green]✓[/green] Monitoring started\n")

    def stop(self):
        """Stop monitoring"""
        console.print("\n[yellow]Stopping monitor...[/yellow]")

        self.running = False

        if self.capture_thread:
            self.capture_thread.stop()

        if self.processing_thread:
            self.processing_thread.stop()

        if self.visualization_process:
            self.stop_visualization()

        console.print("[green]✓[/green] Monitor stopped")

    def generate_display(self) -> Layout:
        """Generate Rich layout for display"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=7)
        )

        layout["body"].split_row(
            Layout(name="metrics"),
            Layout(name="status")
        )

        # Header
        header_text = Text("IR-UWB RESPIRATION MONITOR", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, box=box.DOUBLE))

        # Metrics panel
        metrics_table = self.create_metrics_table()
        layout["metrics"].update(Panel(metrics_table, title="[bold]Breathing Metrics[/bold]", border_style="green"))

        # Status panel
        status_table = self.create_status_table()
        layout["status"].update(Panel(status_table, title="[bold]System Status[/bold]", border_style="blue"))

        # Footer with recent measurements
        history_table = self.create_history_table()
        layout["footer"].update(Panel(history_table, title="[bold]Recent Measurements[/bold]", border_style="cyan"))

        return layout

    def create_metrics_table(self) -> Table:
        """Create metrics display table"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="bold green", width=20)

        current_rate = self.metrics_tracker.get_current_rate()
        avg_rate = self.metrics_tracker.get_average_rate()
        variability = self.metrics_tracker.get_rate_variability()

        # Color code based on rate
        if current_rate == 0:
            rate_style = "dim"
        elif 10 <= current_rate <= 25:
            rate_style = "bold green"
        else:
            rate_style = "bold yellow"

        table.add_row("Current Rate", f"[{rate_style}]{current_rate:.1f} BPM[/{rate_style}]")
        table.add_row("Average (10s)", f"{avg_rate:.1f} BPM")
        table.add_row("Variability", f"{variability:.1f} BPM")

        # Add visual bar for current rate
        if current_rate > 0:
            bar_length = int(min(current_rate / 30 * 20, 20))
            bar = "█" * bar_length
            table.add_row("Rate Visual", f"[green]{bar}[/green]")

        # Latest measurement details
        if self.latest_results:
            table.add_row("", "")
            table.add_row("Chest Range", f"{self.latest_results['chest_range']:.2f} m")
            table.add_row("SNR", f"{self.latest_results['snr']:.1f} dB")
            table.add_row("Quality", f"{self.latest_results['quality_score']:.3f}")

        return table

    def create_status_table(self) -> Table:
        """Create system status table"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Component", style="cyan", width=15)
        table.add_column("Status", width=25)

        # Capture status
        if self.capture_thread:
            capture_status = self.capture_thread.status_message
            capture_style = "green" if "Running" in capture_status else "yellow"
            table.add_row("Capture", f"[{capture_style}]{capture_status}[/{capture_style}]")
            table.add_row("Pulses", f"{self.capture_thread.pulse_count:,}")
        else:
            table.add_row("Capture", "[red]Not started[/red]")

        # Processing status
        if self.processing_thread:
            proc_status = self.processing_thread.status_message
            proc_style = "green" if proc_status == "Ready" else "yellow"
            table.add_row("Processing", f"[{proc_style}]{proc_status}[/{proc_style}]")
        else:
            table.add_row("Processing", "[red]Not started[/red]")

        # Buffer status
        buffer_size = self.data_buffer.get_buffer_size()
        buffer_pct = int(buffer_size / self.data_buffer.max_pulses * 100)
        table.add_row("Buffer", f"{buffer_size}/{self.data_buffer.max_pulses} ({buffer_pct}%)")

        # Visualization status
        viz_status = "Running" if self.visualization_process else "Off"
        viz_style = "green" if self.visualization_process else "dim"
        table.add_row("Visualization", f"[{viz_style}]{viz_status}[/{viz_style}]")

        return table

    def create_history_table(self) -> Table:
        """Create recent measurements table"""
        table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1))
        table.add_column("Time Ago", style="cyan", width=10)
        table.add_column("Rate (BPM)", style="green", width=12, justify="right")
        table.add_column("SNR (dB)", style="yellow", width=10, justify="right")

        trend_data = self.metrics_tracker.get_trend_data()
        recent_rates = list(trend_data['breathing_rates'])[-5:]
        recent_snr = list(trend_data['snr_values'])[-5:]

        for i, (rate, snr) in enumerate(zip(reversed(recent_rates), reversed(recent_snr))):
            ago = (len(recent_rates) - i - 1) * 2
            table.add_row(f"{ago}s ago", f"{rate:.1f}", f"{snr:.1f}")

        return table

    def run_interactive_loop(self):
        """Main interactive loop with live display"""
        console.print("\n[bold cyan]Interactive Commands:[/bold cyan]")
        console.print("  [v] Visualization  [p] Pause  [r] Resume  [e] Export  [x] Clear  [q] Quit\n")

        with Live(self.generate_display(), console=console, refresh_per_second=2) as live:
            try:
                while self.running:
                    # Update latest results
                    try:
                        self.latest_results = self.results_queue.get(timeout=0.5)

                        # Send to visualization if enabled
                        if self.viz_queue is not None:
                            try:
                                self.viz_queue.put_nowait(self.latest_results)
                            except:
                                pass  # Queue full, skip
                    except Empty:
                        pass

                    # Update display
                    live.update(self.generate_display())

            except KeyboardInterrupt:
                pass

    def handle_command(self, command):
        """Handle user command"""
        command = command.lower().strip()

        if command == 'q' or command == 'quit':
            return False  # Signal to quit

        elif command == 'p' or command == 'pause':
            if self.capture_thread:
                self.capture_thread.pause()
                console.print("[yellow]✓[/yellow] Capture paused")

        elif command == 'r' or command == 'resume':
            if self.capture_thread:
                self.capture_thread.resume()
                console.print("[green]✓[/green] Capture resumed")

        elif command == 'v' or command == 'visualize':
            self.toggle_visualization()

        elif command == 'e' or command == 'export':
            self.export_data()

        elif command == 'x' or command == 'clear':
            self.data_buffer.clear()
            console.print("[green]✓[/green] Buffer cleared")

        elif command == 's' or command == 'status':
            self.show_detailed_status()

        elif command == 'h' or command == 'help':
            self.show_help()

        else:
            if command:
                console.print(f"[red]Unknown command: '{command}'[/red]")
                console.print("Type 'h' for help")

        return True  # Continue running

    def toggle_visualization(self):
        """Toggle live visualization"""
        if self.visualization_process:
            self.stop_visualization()
        else:
            self.start_visualization()

    def start_visualization(self):
        """Start live visualization in separate process"""
        try:
            import multiprocessing as mp
            from simple_visualization import run_visualization_standalone

            console.print("[cyan]Starting visualization...[/cyan]")

            # Create shared queue for visualization
            viz_queue = mp.Queue()

            # Start visualization process
            self.visualization_process = mp.Process(
                target=run_visualization_standalone,
                args=(viz_queue,),
                daemon=True
            )
            self.visualization_process.start()

            # Store queue so we can send data to it
            self.viz_queue = viz_queue

            console.print("[green]✓[/green] Visualization started")
            console.print("[dim]  Close the plot window or use 'v' to toggle off[/dim]")
        except Exception as e:
            console.print(f"[red]✗ Visualization error: {e}[/red]")
            console.print("[yellow]Tip: Close any existing matplotlib windows and try again[/yellow]")
            import traceback
            traceback.print_exc()

    def stop_visualization(self):
        """Stop visualization"""
        if self.visualization_process:
            self.visualization_process.terminate()
            self.visualization_process.join(timeout=2.0)
            self.visualization_process = None
            self.viz_queue = None
            console.print("[green]✓[/green] Visualization stopped")

    def export_data(self):
        """Export current data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.OUTPUT_DIR}/export_{timestamp}.npz"

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
            console.print(f"[green]✓[/green] Data exported to: {filename}")
        else:
            console.print("[red]✗[/red] No data to export")

    def show_detailed_status(self):
        """Show detailed system status"""
        console.print("\n[bold]Detailed System Status[/bold]")
        console.print("─" * 60)

        # Capture thread
        if self.capture_thread:
            console.print(f"Capture: {self.capture_thread.status_message}")
            console.print(f"  Pulses: {self.capture_thread.pulse_count:,}")
            console.print(f"  Paused: {self.capture_thread.paused}")

        # Processing thread
        if self.processing_thread:
            console.print(f"Processing: {self.processing_thread.status_message}")
            if self.processing_thread.last_error:
                console.print(f"  [red]Last Error: {self.processing_thread.last_error}[/red]")

        # Buffer
        console.print(f"Buffer: {self.data_buffer.get_buffer_size()}/{self.data_buffer.max_pulses}")

        # Metrics
        console.print(f"Measurements: {len(self.metrics_tracker.breathing_rates)}")

        console.print("─" * 60 + "\n")

    def show_help(self):
        """Show help message"""
        console.print("\n[bold cyan]Available Commands:[/bold cyan]")
        console.print("  [v] visualize - Toggle live visualization (6 plots)")
        console.print("  [p] pause     - Pause data capture")
        console.print("  [r] resume    - Resume data capture")
        console.print("  [e] export    - Export data to .npz file")
        console.print("  [x] clear     - Clear data buffer")
        console.print("  [s] status    - Show detailed status")
        console.print("  [h] help      - Show this help")
        console.print("  [q] quit      - Exit application\n")


def main():
    """Main entry point with proper command loop"""
    console.print("[bold blue]IR-UWB Real-Time Respiration Monitor[/bold blue]")
    console.print("Using Rich CLI for better terminal experience\n")

    # Check hardware
    try:
        import SoapySDR
        console.print("[green]✓[/green] SoapySDR detected")
        use_hw = Confirm.ask("Use hardware?", default=False)
    except ImportError:
        console.print("[yellow]⚠[/yellow] SoapySDR not installed - using synthetic mode")
        use_hw = False

    # Create and start monitor
    monitor = RichCLIMonitor(use_hardware=use_hw)
    monitor.start()

    # Run display in background thread
    import threading
    display_thread = threading.Thread(target=monitor.run_interactive_loop, daemon=True)
    display_thread.start()

    # Command loop
    try:
        while monitor.running:
            command = Prompt.ask("\n[bold cyan]Command[/bold cyan]", default="")

            if not monitor.handle_command(command):
                break  # User quit

    except KeyboardInterrupt:
        console.print("\n[yellow]Ctrl+C detected[/yellow]")

    finally:
        monitor.stop()
        console.print("\n[bold green]Thank you for using IR-UWB Monitor![/bold green]\n")


if __name__ == "__main__":
    main()
