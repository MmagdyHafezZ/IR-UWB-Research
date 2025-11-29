#!/usr/bin/env python3
"""
IR-UWB BPM Monitoring GUI
Real-time graphical interface for breathing rate detection
Fixed version with proper demo mode and data flow
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import queue
import time
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import Config
from range_time_matrix import RangeTimeMatrix
from uwb_processing_fixed import UWBProcessor


class DataBuffer:
    """Thread-safe buffer for radar data"""
    def __init__(self, max_pulses=20000):
        self.max_pulses = max_pulses
        self.buffer = deque(maxlen=max_pulses)
        self.lock = threading.Lock()

    def add_pulse(self, pulse_data):
        with self.lock:
            self.buffer.append(pulse_data)

    def get_data_matrix(self):
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return np.array(list(self.buffer))

    def get_size(self):
        with self.lock:
            return len(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()


class BPMMonitorGUI:
    """Main GUI Application for BPM Monitoring"""

    def __init__(self, root):
        self.root = root
        self.root.title("IR-UWB Breathing Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')

        # System state
        self.is_running = False
        self.use_hardware = False
        self.current_bpm = 0.0
        self.avg_bpm = 0.0
        self.chest_range = 0.0
        self.signal_quality = "Unknown"
        self.snr_db = -10.0

        # Data storage
        self.data_buffer = DataBuffer(max_pulses=20000)
        self.bpm_history = deque(maxlen=60)
        self.breathing_waveform = deque(maxlen=500)
        self.time_history = deque(maxlen=60)

        # Demo mode state - track phase continuity
        self.demo_start_time = 0
        self.demo_bpm = 15.0

        # Processing window
        self.window_pulses = 5000

        # Threading
        self.capture_thread = None
        self.processing_thread = None
        self.results_queue = queue.Queue()

        # UWB Processor
        self.processor = UWBProcessor(Config)

        # Build GUI
        self.setup_styles()
        self.create_widgets()
        self.setup_plots()

        # Start update loop
        self.root.after(100, self.update_display)

    def setup_styles(self):
        """Configure GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')

        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#00ff41',
            'warning': '#ff9800',
            'error': '#f44336',
            'panel': '#2d2d30',
            'button': '#3c3c3c'
        }

    def create_widgets(self):
        """Create all GUI widgets"""
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_bpm_display(main_frame)
        self.create_plot_area(main_frame)
        self.create_control_panel(main_frame)

    def create_bpm_display(self, parent):
        """Create large BPM display panel"""
        bpm_frame = tk.Frame(parent, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        bpm_frame.pack(fill=tk.X, pady=(0, 10))

        # Main BPM Value
        bpm_container = tk.Frame(bpm_frame, bg=self.colors['panel'])
        bpm_container.pack(side=tk.LEFT, padx=20, pady=15)

        tk.Label(bpm_container, text="CURRENT BPM",
                 font=('Arial', 12, 'bold'),
                 bg=self.colors['panel'],
                 fg=self.colors['accent']).pack()

        self.bpm_label = tk.Label(
            bpm_container,
            text="--.-",
            font=('Arial', 72, 'bold'),
            bg=self.colors['panel'],
            fg=self.colors['accent']
        )
        self.bpm_label.pack()

        # Stats Panel
        stats_frame = tk.Frame(bpm_frame, bg=self.colors['panel'])
        stats_frame.pack(side=tk.LEFT, padx=40, pady=15)

        avg_container = tk.Frame(stats_frame, bg=self.colors['panel'])
        avg_container.pack(anchor=tk.W, pady=5)
        tk.Label(avg_container, text="Average:",
                 font=('Arial', 11),
                 bg=self.colors['panel'],
                 fg='#888888').pack(side=tk.LEFT)
        self.avg_label = tk.Label(avg_container,
                                  text="--.- BPM",
                                  font=('Arial', 11, 'bold'),
                                  bg=self.colors['panel'],
                                  fg=self.colors['fg'])
        self.avg_label.pack(side=tk.LEFT, padx=(10, 0))

        range_container = tk.Frame(stats_frame, bg=self.colors['panel'])
        range_container.pack(anchor=tk.W, pady=5)
        tk.Label(range_container, text="Range:",
                 font=('Arial', 11),
                 bg=self.colors['panel'],
                 fg='#888888').pack(side=tk.LEFT)
        self.range_label = tk.Label(range_container,
                                    text="-.-- m",
                                    font=('Arial', 11, 'bold'),
                                    bg=self.colors['panel'],
                                    fg=self.colors['fg'])
        self.range_label.pack(side=tk.LEFT, padx=(10, 0))

        snr_container = tk.Frame(stats_frame, bg=self.colors['panel'])
        snr_container.pack(anchor=tk.W, pady=5)
        tk.Label(snr_container, text="SNR:",
                 font=('Arial', 11),
                 bg=self.colors['panel'],
                 fg='#888888').pack(side=tk.LEFT)
        self.snr_label = tk.Label(snr_container,
                                  text="-- dB",
                                  font=('Arial', 11, 'bold'),
                                  bg=self.colors['panel'],
                                  fg=self.colors['fg'])
        self.snr_label.pack(side=tk.LEFT, padx=(10, 0))

        # Quality Indicator
        quality_frame = tk.Frame(bpm_frame, bg=self.colors['panel'])
        quality_frame.pack(side=tk.RIGHT, padx=20, pady=15)

        tk.Label(quality_frame, text="SIGNAL QUALITY",
                 font=('Arial', 10),
                 bg=self.colors['panel'],
                 fg='#888888').pack()

        self.quality_canvas = tk.Canvas(
            quality_frame,
            width=150,
            height=80,
            bg=self.colors['panel'],
            highlightthickness=0
        )
        self.quality_canvas.pack(pady=5)
        self.draw_quality_meter()

    def create_plot_area(self, parent):
        """Create matplotlib plot area"""
        plot_frame = tk.Frame(parent, bg=self.colors['panel'])
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.fig = Figure(figsize=(12, 4), facecolor=self.colors['panel'])

        self.ax_wave = self.fig.add_subplot(121)
        self.ax_wave.set_title('Breathing Waveform', color=self.colors['fg'], fontsize=10)
        self.ax_wave.set_xlabel('Time (s)', color=self.colors['fg'], fontsize=9)
        self.ax_wave.set_ylabel('Amplitude', color=self.colors['fg'], fontsize=9)
        self.ax_wave.grid(True, alpha=0.2, color=self.colors['fg'])
        self.ax_wave.set_facecolor('#2d2d30')
        self.ax_wave.tick_params(colors=self.colors['fg'], labelsize=8)

        self.ax_history = self.fig.add_subplot(122)
        self.ax_history.set_title('BPM History', color=self.colors['fg'], fontsize=10)
        self.ax_history.set_xlabel('Time (s)', color=self.colors['fg'], fontsize=9)
        self.ax_history.set_ylabel('BPM', color=self.colors['fg'], fontsize=9)
        self.ax_history.grid(True, alpha=0.2, color=self.colors['fg'])
        self.ax_history.set_facecolor('#2d2d30')
        self.ax_history.tick_params(colors=self.colors['fg'], labelsize=8)
        self.ax_history.set_ylim([0, 40])

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self, parent):
        """Create control buttons and status panel"""
        control_frame = tk.Frame(parent, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X)

        button_frame = tk.Frame(control_frame, bg=self.colors['panel'])
        button_frame.pack(side=tk.LEFT, padx=20, pady=15)

        self.start_btn = tk.Button(
            button_frame,
            text="▶ START",
            command=self.start_monitoring,
            font=('Arial', 12, 'bold'),
            bg=self.colors['accent'],
            fg=self.colors['bg'],
            width=12,
            height=2,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.NORMAL
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            button_frame,
            text="◼ STOP",
            command=self.stop_monitoring,
            font=('Arial', 12, 'bold'),
            bg=self.colors['error'],
            fg=self.colors['fg'],
            width=12,
            height=2,
            relief=tk.FLAT,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        mode_frame = tk.Frame(control_frame, bg=self.colors['panel'])
        mode_frame.pack(side=tk.LEFT, padx=30, pady=15)

        tk.Label(mode_frame, text="Mode:",
                 font=('Arial', 11),
                 bg=self.colors['panel'],
                 fg=self.colors['fg']).pack(side=tk.LEFT)

        self.mode_var = tk.StringVar(value="demo")
        tk.Radiobutton(mode_frame, text="Demo",
                       variable=self.mode_var, value="demo",
                       font=('Arial', 10),
                       bg=self.colors['panel'],
                       fg=self.colors['fg'],
                       selectcolor=self.colors['panel'],
                       activebackground=self.colors['panel']).pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(mode_frame, text="Hardware",
                       variable=self.mode_var, value="hardware",
                       font=('Arial', 10),
                       bg=self.colors['panel'],
                       fg=self.colors['fg'],
                       selectcolor=self.colors['panel'],
                       activebackground=self.colors['panel']).pack(side=tk.LEFT)

        status_frame = tk.Frame(control_frame, bg=self.colors['panel'])
        status_frame.pack(side=tk.RIGHT, padx=20, pady=15)

        tk.Label(status_frame, text="System Status",
                 font=('Arial', 11, 'bold'),
                 bg=self.colors['panel'],
                 fg=self.colors['fg']).pack(anchor=tk.E)

        self.status_label = tk.Label(status_frame,
                                     text="● IDLE",
                                     font=('Arial', 10),
                                     bg=self.colors['panel'],
                                     fg=self.colors['warning'])
        self.status_label.pack(anchor=tk.E, pady=2)

        self.buffer_label = tk.Label(status_frame,
                                     text="Buffer: 0/20000",
                                     font=('Arial', 9),
                                     bg=self.colors['panel'],
                                     fg='#888888')
        self.buffer_label.pack(anchor=tk.E)

    def setup_plots(self):
        """Initialize plot lines"""
        self.line_wave, = self.ax_wave.plot([], [], 'g-', linewidth=2)
        self.line_history, = self.ax_history.plot([], [], 'b-', marker='o', markersize=4)

        self.ax_history.axhline(y=15, color='g', linestyle='--', alpha=0.3, label='Normal')
        self.ax_history.legend(loc='upper right', fontsize=8)

    def draw_quality_meter(self):
        """Draw signal quality meter"""
        self.quality_canvas.delete("all")

        x, y, w, h = 10, 20, 130, 40
        self.quality_canvas.create_rectangle(
            x, y, x + w, y + h,
            fill='#1e1e1e',
            outline='#444444'
        )

        quality_colors = {
            'Poor': '#f44336',
            'Fair': '#ff9800',
            'Good': '#4caf50',
            'Excellent': '#00ff41'
        }

        if self.signal_quality in quality_colors:
            color = quality_colors[self.signal_quality]
            if self.signal_quality == 'Poor':
                fill_width = w * 0.25
            elif self.signal_quality == 'Fair':
                fill_width = w * 0.5
            elif self.signal_quality == 'Good':
                fill_width = w * 0.75
            else:
                fill_width = w

            self.quality_canvas.create_rectangle(
                x, y, x + fill_width, y + h,
                fill=color,
                outline=''
            )

        self.quality_canvas.create_text(
            x + w / 2,
            y + h / 2,
            text=self.signal_quality.upper(),
            fill=self.colors['fg'],
            font=('Arial', 10, 'bold')
        )

    def start_monitoring(self):
        """Start BPM monitoring"""
        if self.is_running:
            return

        self.is_running = True
        self.use_hardware = (self.mode_var.get() == "hardware")
        self.demo_start_time = time.time()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="● RUNNING", fg=self.colors['accent'])

        self.data_buffer.clear()
        self.bpm_history.clear()
        self.breathing_waveform.clear()
        self.time_history.clear()

        self.capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
        self.capture_thread.start()

        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

    def stop_monitoring(self):
        """Stop BPM monitoring"""
        self.is_running = False

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="● STOPPED", fg=self.colors['error'])

    def capture_worker(self):
        """Worker thread for data capture"""
        capture = None

        if self.use_hardware:
            try:
                from sdr_capture import SDRCapture
                capture = SDRCapture()
                capture.setup_transmitter()
                capture.setup_receiver()
                capture.generate_impulse_signal()
            except Exception as e:
                print(f"Hardware init failed, falling back to demo: {e}")
                self.use_hardware = False
                capture = None

        if self.use_hardware and capture is not None:
            while self.is_running:
                try:
                    pulse = capture.record_single_pulse()
                    if pulse is not None and len(pulse) > 0:
                        self.data_buffer.add_pulse(pulse)
                except Exception as e:
                    print(f"Capture error: {e}")
                    time.sleep(0.01)
        else:
            # Synthetic data generation (demo mode)
            samples_per_pulse = Config.SAMPLES_PER_PULSE
            pulse_count = 0

            while self.is_running:
                try:
                    t = pulse_count * (1.0 / Config.PULSE_REPETITION_FREQ)
                    breathing_phase = 2 * np.pi * (self.demo_bpm / 60.0) * t

                    pulse = np.zeros(samples_per_pulse, dtype=complex)
                    chest_bin = 100
                    pulse[chest_bin] = np.exp(1j * breathing_phase)

                    pulse += 0.1 * (
                        np.random.randn(samples_per_pulse) +
                        1j * np.random.randn(samples_per_pulse)
                    )

                    self.data_buffer.add_pulse(pulse)
                    pulse_count += 1

                    time.sleep(1.0 / Config.PULSE_REPETITION_FREQ)
                except Exception as e:
                    print(f"Demo capture error: {e}")
                    time.sleep(0.1)

    def processing_worker(self):
        """Worker thread for signal processing"""
        while self.is_running:
            time.sleep(1.0)

            # DEMO MODE: generate continuous synthetic breathing waveform
            if not self.use_hardware:
                try:
                    elapsed = time.time() - self.demo_start_time
                    fs = 50  # Display sample rate
                    n_samples = 500
                    
                    # Generate waveform centered at current time
                    t_end = elapsed
                    t_start = max(0, t_end - n_samples / fs)
                    tt = np.linspace(t_start, t_end, n_samples)
                    
                    # Add slight variation to BPM for realism
                    bpm_variation = self.demo_bpm + 0.5 * np.sin(elapsed * 0.1)
                    breathing_freq = bpm_variation / 60.0
                    waveform = 0.1 * np.sin(2 * np.pi * breathing_freq * tt)
                    
                    # Add small noise
                    waveform += 0.005 * np.random.randn(len(waveform))
                    
                    results = {
                        'bpm': bpm_variation,
                        'chest_range': 1.0,
                        'quality': 'good',
                        'breathing_signal': waveform,
                        'time_axis': tt
                    }
                    self.results_queue.put(results)
                except Exception as e:
                    print(f"Demo processing error: {e}")
                continue

            # HARDWARE MODE: use actual processing chain
            if self.data_buffer.get_size() < max(500, self.window_pulses // 4):
                continue

            raw_data = self.data_buffer.get_data_matrix()
            if raw_data is None:
                continue
            if len(raw_data) > self.window_pulses:
                raw_data = raw_data[-self.window_pulses:]

            try:
                rtm = RangeTimeMatrix(raw_data)
                rtm_matrix = rtm.construct_matrix()
                results = self.processor.process_complete(rtm_matrix)
                self.results_queue.put(results)
            except Exception as e:
                print(f"Processing error: {e}")

    def update_display(self):
        """Update GUI display (called periodically)"""
        try:
            while not self.results_queue.empty():
                results = self.results_queue.get_nowait()
                self.process_results(results)
        except queue.Empty:
            pass

        buffer_size = self.data_buffer.get_size()
        self.buffer_label.config(text=f"Buffer: {buffer_size}/20000")

        self.update_plots()
        self.root.after(100, self.update_display)

    def process_results(self, results):
        """Process and display new results"""
        bpm = results.get('bpm', 0)
        chest_range = results.get('chest_range', 0)
        quality = results.get('quality', 'unknown')
        breathing_signal = results.get('breathing_signal', np.zeros(0))

        self.current_bpm = bpm
        self.chest_range = chest_range

        if bpm > 0:
            self.bpm_history.append(bpm)
            self.time_history.append(time.time())
            self.avg_bpm = float(np.mean(list(self.bpm_history)))

        # Update breathing waveform - replace entirely for smooth display
        if len(breathing_signal) > 0:
            self.breathing_waveform.clear()
            for sample in breathing_signal[-500:]:
                val = float(sample.real) if np.iscomplexobj(sample) else float(sample)
                self.breathing_waveform.append(val)

        if quality.lower() == 'good':
            self.signal_quality = 'Good'
            self.snr_db = 10.0
        elif quality.lower() == 'poor':
            self.signal_quality = 'Fair'
            self.snr_db = 5.0
        else:
            self.signal_quality = 'Poor'
            self.snr_db = 0.0

        self.update_labels()

    def update_labels(self):
        """Update all text labels"""
        if self.current_bpm > 0:
            self.bpm_label.config(text=f"{self.current_bpm:.1f}")

            if 12 <= self.current_bpm <= 20:
                color = self.colors['accent']
            elif 6 <= self.current_bpm <= 42:
                color = self.colors['warning']
            else:
                color = self.colors['error']
            self.bpm_label.config(fg=color)
        else:
            self.bpm_label.config(text="--.-", fg='#888888')

        self.avg_label.config(
            text=f"{self.avg_bpm:.1f} BPM" if self.avg_bpm > 0 else "--- BPM"
        )
        self.range_label.config(
            text=f"{self.chest_range:.2f} m" if self.chest_range > 0 else "-.-- m"
        )
        self.snr_label.config(text=f"{self.snr_db:.1f} dB")

        self.draw_quality_meter()

    def update_plots(self):
        """Update matplotlib plots"""
        # Breathing waveform plot
        if len(self.breathing_waveform) > 0:
            waveform_data = list(self.breathing_waveform)
            n = len(waveform_data)
            time_axis = np.arange(n) / 50.0  # 50 Hz display rate
            
            self.line_wave.set_data(time_axis, waveform_data)
            self.ax_wave.set_xlim(0, max(1, time_axis[-1]))
            
            y_min = min(waveform_data) - 0.05
            y_max = max(waveform_data) + 0.05
            if y_max - y_min < 0.1:
                y_min, y_max = -0.1, 0.1
            self.ax_wave.set_ylim(y_min, y_max)
        else:
            self.line_wave.set_data([], [])
            self.ax_wave.set_xlim(0, 10)
            self.ax_wave.set_ylim(-0.15, 0.15)

        # BPM history plot
        if len(self.bpm_history) > 0 and len(self.time_history) > 0:
            t0 = self.time_history[0]
            rel_time = [t - t0 for t in self.time_history]
            bpm_data = list(self.bpm_history)

            self.line_history.set_data(rel_time, bpm_data)

            if len(rel_time) > 0:
                self.ax_history.set_xlim(
                    max(0, rel_time[-1] - 60),
                    rel_time[-1] + 5
                )
                self.ax_history.set_ylim(
                    0,
                    max(40, max(bpm_data) + 5)
                )
        else:
            self.line_history.set_data([], [])
            self.ax_history.set_xlim(0, 60)
            self.ax_history.set_ylim(0, 40)

        try:
            self.canvas.draw_idle()
        except Exception:
            pass


def main():
    root = tk.Tk()

    try:
        root.iconbitmap('icon.ico')
    except Exception:
        pass

    app = BPMMonitorGUI(root)

    def on_closing():
        app.stop_monitoring()
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    print("Starting IR-UWB BPM Monitor GUI...")
    main()