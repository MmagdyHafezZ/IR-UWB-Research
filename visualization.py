"""
Visualization Module
Provides plotting functions for all stages of the IR-UWB respiration detection pipeline
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import Config


def _show_plot_if_enabled():
    """Show plot only if PLOT_RESULTS is enabled, otherwise just close"""
    if Config.PLOT_RESULTS:
        try:
            plt.show()
        except:
            
            plt.close()
    else:
        plt.close()


class Visualizer:
    """Visualization utilities for IR-UWB data"""

    def __init__(self, output_dir=None):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir if output_dir is not None else Config.OUTPUT_DIR
        plt.style.use('default')

    def plot_raw_data_sample(self, raw_data, num_pulses=10, save=True):
        """
        Plot a sample of raw IQ data

        Args:
            raw_data: Raw IQ data array
            num_pulses: Number of pulses to plot
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        if raw_data.ndim == 2:
            
            for i in range(min(num_pulses, raw_data.shape[0])):
                axes[0].plot(np.abs(raw_data[i, :]), alpha=0.7, label=f'Pulse {i+1}')
                axes[1].plot(np.angle(raw_data[i, :]), alpha=0.7)

            axes[0].set_title('Raw IQ Data - Magnitude')
            axes[0].set_xlabel('Sample Index')
            axes[0].set_ylabel('Magnitude')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].set_title('Raw IQ Data - Phase')
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Phase (radians)')
            axes[1].grid(True)

        else:
            
            sample_length = min(10000, len(raw_data))
            axes[0].plot(np.abs(raw_data[:sample_length]))
            axes[0].set_title('Raw IQ Data - Magnitude')
            axes[0].set_xlabel('Sample Index')
            axes[0].set_ylabel('Magnitude')
            axes[0].grid(True)

            axes[1].plot(np.angle(raw_data[:sample_length]))
            axes[1].set_title('Raw IQ Data - Phase')
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Phase (radians)')
            axes[1].grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/raw_data_sample.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}/raw_data_sample.png")

        _show_plot_if_enabled()

    def plot_range_time_matrix(self, rtm, range_bins=None, time_axis=None, save=True):
        """
        Plot range-time matrix as 2D heatmap

        Args:
            rtm: Range-time matrix (2D complex array)
            range_bins: Range values in meters
            time_axis: Time values in seconds
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        
        magnitude = np.abs(rtm)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        if range_bins is not None and time_axis is not None:
            extent = [range_bins[0], range_bins[-1], time_axis[-1], time_axis[0]]
            im0 = axes[0].imshow(magnitude_db, aspect='auto', cmap='jet', extent=extent)
            axes[0].set_xlabel('Range (m)')
            axes[0].set_ylabel('Time (s)')
        else:
            im0 = axes[0].imshow(magnitude_db, aspect='auto', cmap='jet')
            axes[0].set_xlabel('Range Bin')
            axes[0].set_ylabel('Frame Number')

        axes[0].set_title('Range-Time Matrix - Magnitude (dB)')
        plt.colorbar(im0, ax=axes[0], label='Magnitude (dB)')

        
        phase = np.angle(rtm)

        if range_bins is not None and time_axis is not None:
            im1 = axes[1].imshow(phase, aspect='auto', cmap='hsv', extent=extent)
            axes[1].set_xlabel('Range (m)')
            axes[1].set_ylabel('Time (s)')
        else:
            im1 = axes[1].imshow(phase, aspect='auto', cmap='hsv')
            axes[1].set_xlabel('Range Bin')
            axes[1].set_ylabel('Frame Number')

        axes[1].set_title('Range-Time Matrix - Phase')
        plt.colorbar(im1, ax=axes[1], label='Phase (radians)')

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/range_time_matrix.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}/range_time_matrix.png")

        _show_plot_if_enabled()

    def plot_variance_profile(self, variance_profile, range_bins=None, chest_bin=None, save=True):
        """
        Plot slow-time variance profile across range bins

        Args:
            variance_profile: 1D array of variance values
            range_bins: Range values in meters
            chest_bin: Index of detected chest bin
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if range_bins is not None:
            ax.plot(range_bins, variance_profile, 'b-', linewidth=2)
            ax.set_xlabel('Range (m)')

            if chest_bin is not None:
                ax.axvline(range_bins[chest_bin], color='r', linestyle='--',
                          linewidth=2, label=f'Detected Chest ({range_bins[chest_bin]:.2f} m)')
        else:
            ax.plot(variance_profile, 'b-', linewidth=2)
            ax.set_xlabel('Range Bin')

            if chest_bin is not None:
                ax.axvline(chest_bin, color='r', linestyle='--',
                          linewidth=2, label=f'Detected Chest (bin {chest_bin})')

        ax.set_ylabel('Variance')
        ax.set_title('Slow-Time Variance Profile (Chest Detection)')
        ax.grid(True, alpha=0.3)

        if chest_bin is not None:
            ax.legend()

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/variance_profile.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}/variance_profile.png")

        _show_plot_if_enabled()

    def plot_breathing_waveform(self, time_axis, waveform, breathing_rate=None, save=True):
        """
        Plot breathing waveform

        Args:
            time_axis: Time values in seconds
            waveform: Breathing signal
            breathing_rate: Detected breathing rate in BPM
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(time_axis, waveform, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Chest Displacement (a.u.)')

        if breathing_rate is not None:
            title = f'Breathing Waveform (Rate: {breathing_rate:.1f} BPM)'
        else:
            title = 'Breathing Waveform'

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/breathing_waveform.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}/breathing_waveform.png")

        _show_plot_if_enabled()

    def plot_frequency_spectrum(self, frequencies, spectrum, breathing_rate_freq=None, save=True):
        """
        Plot frequency spectrum of breathing signal

        Args:
            frequencies: Frequency values in Hz
            spectrum: Magnitude spectrum
            breathing_rate_freq: Detected breathing rate in Hz
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        
        mask = (frequencies >= 0) & (frequencies <= 1.0)
        freq_plot = frequencies[mask]
        spec_plot = spectrum[mask]

        ax.plot(freq_plot, spec_plot, 'b-', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('Frequency Spectrum of Breathing Signal')
        ax.grid(True, alpha=0.3)

        
        ax.axvspan(Config.BREATHING_FREQ_MIN, Config.BREATHING_FREQ_MAX,
                  alpha=0.2, color='green', label='Breathing Range')

        
        if breathing_rate_freq is not None:
            ax.axvline(breathing_rate_freq, color='r', linestyle='--',
                      linewidth=2, label=f'Detected Rate ({breathing_rate_freq:.3f} Hz)')

        ax.legend()

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/frequency_spectrum.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}/frequency_spectrum.png")

        _show_plot_if_enabled()

    def plot_complete_analysis(self, rtm, variance_profile, range_bins, time_axis,
                              chest_bin, breathing_waveform, breathing_time,
                              frequencies, spectrum, results, save=True):
        """
        Create comprehensive analysis plot with all results

        Args:
            rtm: Range-time matrix
            variance_profile: Variance profile
            range_bins: Range values
            time_axis: Slow-time axis
            chest_bin: Detected chest bin
            breathing_waveform: Filtered breathing signal
            breathing_time: Time axis for breathing waveform
            frequencies: Frequency values
            spectrum: Magnitude spectrum
            results: Dictionary with analysis results
            save: Whether to save the plot
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        
        ax1 = fig.add_subplot(gs[0, :])
        magnitude = np.abs(rtm)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        extent = [range_bins[0], range_bins[-1], time_axis[-1], time_axis[0]]
        im1 = ax1.imshow(magnitude_db, aspect='auto', cmap='jet', extent=extent)
        ax1.axvline(range_bins[chest_bin], color='white', linestyle='--',
                   linewidth=2, label='Chest Location')
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Range-Time Matrix (Magnitude)')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
        ax1.legend(loc='upper right')

        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(range_bins, variance_profile, 'b-', linewidth=2)
        ax2.axvline(range_bins[chest_bin], color='r', linestyle='--',
                   linewidth=2, label=f'Chest: {range_bins[chest_bin]:.2f} m')
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Slow-Time Variance Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(breathing_time, breathing_waveform, 'b-', linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Chest Displacement (a.u.)')
        ax3.set_title(f'Breathing Waveform ({results["breathing_rate_avg"]:.1f} BPM)')
        ax3.grid(True, alpha=0.3)

        
        ax4 = fig.add_subplot(gs[2, 0])
        mask = (frequencies >= 0) & (frequencies <= 1.0)
        ax4.plot(frequencies[mask], spectrum[mask], 'b-', linewidth=2)
        ax4.axvspan(Config.BREATHING_FREQ_MIN, Config.BREATHING_FREQ_MAX,
                   alpha=0.2, color='green', label='Breathing Range')
        breathing_freq_hz = results['breathing_rate_freq'] / 60
        ax4.axvline(breathing_freq_hz, color='r', linestyle='--',
                   linewidth=2, label=f'{breathing_freq_hz:.3f} Hz')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('Breathing Frequency Spectrum')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        summary_text = f"""
        === RESPIRATION ANALYSIS RESULTS ===

        Breathing Rate (Time-domain): {results['breathing_rate_time']:.1f} BPM
        Breathing Rate (Freq-domain): {results['breathing_rate_freq']:.1f} BPM
        Average Breathing Rate: {results['breathing_rate_avg']:.1f} BPM

        Signal Quality: {results['signal_quality']:.2f}

        Chest Location: {range_bins[chest_bin]:.2f} m (bin {chest_bin})

        === BREATHING PATTERN ===
        Number of Breaths: {results['pattern_metrics']['num_breaths']}
        Breathing Depth: {results['pattern_metrics']['breathing_depth']:.3f}
        Regularity: {results['pattern_metrics']['regularity']:.2f}
        I/E Ratio: {results['pattern_metrics']['ie_ratio']:.2f}

        === SYSTEM PARAMETERS ===
        PRF: {Config.PULSE_REPETITION_FREQ} Hz
        Range Resolution: {Config.get_range_resolution():.4f} m
        Max Range: {Config.get_max_range():.2f} m
        """

        ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if save:
            plt.savefig(f'{self.output_dir}/complete_analysis.png', dpi=150, bbox_inches='tight')
            print(f"Saved comprehensive analysis to {self.output_dir}/complete_analysis.png")

        _show_plot_if_enabled()

    def plot_vmd_modes(self, modes, omega, fs, time_axis=None, selected_mode_idx=None, save=True):
        """
        Plot all VMD modes and their spectra

        Args:
            modes: VMD modes array (K x N)
            omega: Center frequencies of modes (normalized)
            fs: Sampling rate in Hz
            time_axis: Time axis for modes (optional)
            selected_mode_idx: Index of selected respiration mode
            save: Whether to save the plot
        """
        from scipy.fft import fft, fftfreq

        K = modes.shape[0]
        N = modes.shape[1]

        if time_axis is None:
            time_axis = np.arange(N) / fs

        
        fig, axes = plt.subplots(K, 2, figsize=(16, 3*K))

        if K == 1:
            axes = axes.reshape(1, -1)

        for k in range(K):
            
            ax_time = axes[k, 0]
            ax_time.plot(time_axis, modes[k, :], 'b-', linewidth=1)
            ax_time.set_ylabel('Amplitude')
            ax_time.grid(True, alpha=0.3)

            
            if selected_mode_idx is not None and k == selected_mode_idx:
                ax_time.set_title(f'Mode {k} (RESPIRATION) - Center: {omega[k]*fs:.3f} Hz',
                                fontweight='bold', color='red')
                ax_time.patch.set_facecolor('#ffe6e6')
            else:
                ax_time.set_title(f'Mode {k} - Center: {omega[k]*fs:.3f} Hz')

            if k == K - 1:
                ax_time.set_xlabel('Time (s)')

            
            ax_freq = axes[k, 1]
            mode_fft = fft(modes[k, :])
            freqs = fftfreq(N, d=1.0/fs)

            
            pos_idx = freqs > 0
            freqs_pos = freqs[pos_idx]
            power = np.abs(mode_fft[pos_idx])**2

            
            mask = freqs_pos <= 1.0
            ax_freq.plot(freqs_pos[mask], power[mask], 'g-', linewidth=2)

            
            ax_freq.axvspan(Config.BREATHING_FREQ_MIN, Config.BREATHING_FREQ_MAX,
                           alpha=0.2, color='yellow', label='Breathing Range')

            
            ax_freq.axvline(omega[k]*fs, color='r', linestyle='--',
                          linewidth=1, label=f'Center: {omega[k]*fs:.3f} Hz')

            ax_freq.set_ylabel('Power')
            ax_freq.grid(True, alpha=0.3)
            ax_freq.legend(fontsize=8)

            if k == 0:
                ax_freq.set_title('Power Spectrum')
            if k == K - 1:
                ax_freq.set_xlabel('Frequency (Hz)')

            
            if selected_mode_idx is not None and k == selected_mode_idx:
                ax_freq.patch.set_facecolor('#ffe6e6')

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/vmd_modes.png', dpi=150, bbox_inches='tight')
            print(f"Saved VMD modes plot to {self.output_dir}/vmd_modes.png")

        _show_plot_if_enabled()

    def plot_vmd_mode_comparison(self, mode_info, save=True):
        """
        Plot comparison of VMD mode characteristics

        Args:
            mode_info: List of dictionaries with mode information
            save: Whether to save the plot
        """
        K = len(mode_info)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        
        mode_indices = [info['mode_index'] for info in mode_info]
        center_freqs = [info['center_freq'] for info in mode_info]
        peak_freqs = [info['peak_freq'] for info in mode_info]
        total_powers = [info['total_power'] for info in mode_info]
        breathing_powers = [info['breathing_power'] for info in mode_info]
        breathing_ratios = [info['breathing_power_ratio'] for info in mode_info]

        
        axes[0, 0].bar(mode_indices, center_freqs, color='steelblue', alpha=0.7)
        axes[0, 0].axhspan(Config.BREATHING_FREQ_MIN, Config.BREATHING_FREQ_MAX,
                          alpha=0.2, color='green', label='Breathing Range')
        axes[0, 0].set_xlabel('Mode Index')
        axes[0, 0].set_ylabel('Center Frequency (Hz)')
        axes[0, 0].set_title('Mode Center Frequencies')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].legend()

        
        axes[0, 1].bar(mode_indices, total_powers, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Mode Index')
        axes[0, 1].set_ylabel('Total Power')
        axes[0, 1].set_title('Mode Total Power')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        
        axes[1, 0].bar(mode_indices, breathing_powers, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Mode Index')
        axes[1, 0].set_ylabel('Breathing Band Power')
        axes[1, 0].set_title('Power in Breathing Frequency Range')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        
        axes[1, 1].bar(mode_indices, breathing_ratios, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Mode Index')
        axes[1, 1].set_ylabel('Breathing Power Ratio')
        axes[1, 1].set_title('Fraction of Power in Breathing Band')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/vmd_mode_comparison.png', dpi=150, bbox_inches='tight')
            print(f"Saved VMD mode comparison to {self.output_dir}/vmd_mode_comparison.png")

        _show_plot_if_enabled()

    def plot_baseline_vs_vmd_comparison(self, baseline_waveform, vmd_waveform,
                                       time_axis, baseline_rate, vmd_rate, save=True):
        """
        Plot side-by-side comparison of baseline vs VMD breathing extraction

        Args:
            baseline_waveform: Breathing signal from bandpass filter
            vmd_waveform: Breathing signal from VMD mode
            time_axis: Time axis
            baseline_rate: Breathing rate from baseline method (BPM)
            vmd_rate: Breathing rate from VMD method (BPM)
            save: Whether to save the plot
        """
        from scipy.fft import fft, fftfreq

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        
        axes[0, 0].plot(time_axis, baseline_waveform, 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Chest Displacement (a.u.)')
        axes[0, 0].set_title(f'Baseline (Bandpass Filter) - {baseline_rate:.1f} BPM')
        axes[0, 0].grid(True, alpha=0.3)

        
        axes[0, 1].plot(time_axis, vmd_waveform, 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Chest Displacement (a.u.)')
        axes[0, 1].set_title(f'VMD - {vmd_rate:.1f} BPM')
        axes[0, 1].grid(True, alpha=0.3)

        
        N = len(baseline_waveform)
        fs = 1.0 / (time_axis[1] - time_axis[0])
        baseline_fft = fft(baseline_waveform)
        freqs = fftfreq(N, d=1.0/fs)
        pos_idx = freqs > 0
        freqs_pos = freqs[pos_idx]
        baseline_power = np.abs(baseline_fft[pos_idx])**2

        mask = freqs_pos <= 1.0
        axes[1, 0].plot(freqs_pos[mask], baseline_power[mask], 'b-', linewidth=2)
        axes[1, 0].axvspan(Config.BREATHING_FREQ_MIN, Config.BREATHING_FREQ_MAX,
                          alpha=0.2, color='green', label='Breathing Range')
        axes[1, 0].axvline(baseline_rate/60, color='r', linestyle='--',
                          linewidth=2, label=f'{baseline_rate/60:.3f} Hz')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].set_title('Baseline Spectrum')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        
        vmd_fft = fft(vmd_waveform)
        vmd_power = np.abs(vmd_fft[pos_idx])**2

        axes[1, 1].plot(freqs_pos[mask], vmd_power[mask], 'r-', linewidth=2)
        axes[1, 1].axvspan(Config.BREATHING_FREQ_MIN, Config.BREATHING_FREQ_MAX,
                          alpha=0.2, color='green', label='Breathing Range')
        axes[1, 1].axvline(vmd_rate/60, color='r', linestyle='--',
                          linewidth=2, label=f'{vmd_rate/60:.3f} Hz')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_title('VMD Spectrum')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.suptitle(f'Baseline vs VMD Comparison (Difference: {abs(baseline_rate - vmd_rate):.1f} BPM)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/baseline_vs_vmd.png', dpi=150, bbox_inches='tight')
            print(f"Saved baseline vs VMD comparison to {self.output_dir}/baseline_vs_vmd.png")

        _show_plot_if_enabled()


def test_visualizer():
    """Test function for visualizer"""
    print("Testing Visualizer Module")
    print("=" * 60)

    
    num_pulses = 500
    samples_per_pulse = 300
    target_bin = 150

    
    rtm = np.zeros((num_pulses, samples_per_pulse), dtype=np.complex64)

    time_axis = np.arange(num_pulses) / Config.PULSE_REPETITION_FREQ
    breathing_freq = 0.3  
    phase_mod = 0.5 * np.sin(2 * np.pi * breathing_freq * time_axis)

    for i in range(num_pulses):
        rtm[i, target_bin] = np.exp(1j * phase_mod[i])

    
    rtm += 0.1 * (np.random.randn(num_pulses, samples_per_pulse) +
                  1j * np.random.randn(num_pulses, samples_per_pulse))

    
    range_bins = np.linspace(0, 10, samples_per_pulse)

    
    import os
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    vis = Visualizer(Config.OUTPUT_DIR)

    
    print("Plotting range-time matrix...")
    vis.plot_range_time_matrix(rtm, range_bins, time_axis, save=True)

    print("Test complete")


if __name__ == "__main__":
    test_visualizer()
