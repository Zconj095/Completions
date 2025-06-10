import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Union
import warnings
from decimal import Decimal, getcontext
from collections import deque
import threading
import time
import random
from threading import Lock
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BrainwaveAnalyzer:
    """
    FIXED VERSION: Proper window positioning and management to prevent stacking and freezing.
    """
    
    def __init__(self, single_window_mode=False):
        # Set precision for 11 decimal places
        getcontext().prec = 15
        self.speed_of_light = Decimal('299792458')
        self.max_frequencies = 5000
        
        # Window management
        self.single_window_mode = single_window_mode
        
        # Thread locks for data safety
        self.data_locks = {}
        
        # Real-time data storage for each brainwave band
        self.live_data = {}
        for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
            self.live_data[band] = {
                'frequencies': deque(maxlen=1000),
                'wavelengths': deque(maxlen=1000),
                'amplitudes': deque(maxlen=1000),
                'timestamps': deque(maxlen=1000)
            }
            self.data_locks[band] = Lock()
        
        # Animation control
        self.is_running = False
        self.data_thread = None
        self.figs = {}
        self.axes = {}
        self.lines = {}
        self.animations = {}
        
        # Performance tracking
        self.update_count = 0
        self.start_time = None
        
    # Constants
    FREQUENCY_RANGES = {
        'Delta': (Decimal('0.50000000000'), Decimal('4.00000000000')),
        'Theta': (Decimal('4.00000000000'), Decimal('8.00000000000')),
        'Alpha': (Decimal('8.00000000000'), Decimal('13.00000000000')),
        'Beta': (Decimal('13.00000000000'), Decimal('30.00000000000')),
        'Gamma': (Decimal('30.00000000000'), Decimal('100.00000000000'))
    }
    
    CORTICAL_ASSOCIATIONS = {
        'Alpha': ['Occipital Lobe', 'Parietal Lobe'],
        'Beta': ['Frontal Lobe', 'Temporal Lobe'],
        'Theta': ['Temporal Lobe', 'Parietal Lobe'],
        'Delta': ['Frontal Lobe', 'Occipital Lobe'],
        'Gamma': ['All Lobes']
    }
    
    ACTIVITY_ASSOCIATIONS = {
        'Alpha': ['Relaxation', 'Reduced anxiety', 'Creativity', 'Wakeful rest'],
        'Beta': ['Alertness', 'Concentration', 'Problem-solving', 'Active thinking'],
        'Theta': ['Deep relaxation', 'Daydreaming', 'Meditation', 'Memory consolidation'],
        'Delta': ['Deep sleep', 'Unconsciousness', 'Healing', 'Regeneration'],
        'Gamma': ['Enhanced sensory processing', 'Information binding', 'Consciousness']
    }
    
    BAND_COLORS = {
        'Delta': '#3498db',    # Blue
        'Theta': '#2ecc71',    # Green  
        'Alpha': '#f39c12',    # Orange
        'Beta': '#e74c3c',     # Red
        'Gamma': '#9b59b6'     # Purple
    }
    
    def simulate_live_data(self):
        """Simulate live brainwave data generation for all bands - THREAD SAFE."""
        start_time = time.time()
        
        while self.is_running:
            current_time = time.time() - start_time
            
            # Generate data for each brainwave band
            for band, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
                min_f = float(min_freq)
                max_f = float(max_freq)
                
                # Generate frequency within band range with some variation
                base_freq = (min_f + max_f) / 2
                variation = (max_f - min_f) * 0.3
                noise = random.uniform(-variation, variation)
                frequency = max(min_f, min(max_f - 0.1, base_freq + noise + 
                                         np.sin(current_time * 0.5) * variation * 0.2))
                
                # Calculate corresponding values
                wavelength = float(self.calculate_high_precision_wavelength(frequency))
                amplitude = 0.5 + 0.3 * np.sin(current_time * 1.5 + hash(band) % 10) + 0.1 * random.uniform(-1, 1)
                amplitude = max(0.1, amplitude)  # Ensure positive amplitude
                
                # THREAD SAFE: Add to live data for this band
                with self.data_locks[band]:
                    self.live_data[band]['frequencies'].append(frequency)
                    self.live_data[band]['wavelengths'].append(wavelength)
                    self.live_data[band]['amplitudes'].append(amplitude)
                    self.live_data[band]['timestamps'].append(current_time)
            
            time.sleep(0.05)  # 20 Hz update rate
    
    def get_band_data_safe(self, band):
        """Safely get data for a band - prevents race conditions."""
        with self.data_locks[band]:
            if len(self.live_data[band]['frequencies']) < 2:
                return None, None, None, None
            
            # Get current data for this band - ensure all arrays have same length
            min_length = min(
                len(self.live_data[band]['timestamps']),
                len(self.live_data[band]['frequencies']),
                len(self.live_data[band]['wavelengths']),
                len(self.live_data[band]['amplitudes'])
            )
            
            times = list(self.live_data[band]['timestamps'])[-min_length:]
            freqs = list(self.live_data[band]['frequencies'])[-min_length:]
            waves = list(self.live_data[band]['wavelengths'])[-min_length:]
            amps = list(self.live_data[band]['amplitudes'])[-min_length:]
            
            return times, freqs, waves, amps
    
    def setup_single_window_visualization(self):
        """Setup a single window with all bands in subplots - PREVENTS WINDOW STACKING."""
        plt.style.use('dark_background')
        
        # Create one large figure with subplots for all bands
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Real-Time Brainwave Analysis Dashboard - All Bands', fontsize=16, color='white')
        
        # Position the window
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+100+50")  # Position at x=100, y=50
        
        self.figs['combined'] = fig
        self.axes = {}
        self.lines = {}
        
        # Create subplots - 5 rows (one per band), 4 columns (freq, wave, amp, stats)
        bands = list(self.FREQUENCY_RANGES.keys())
        
        for i, band in enumerate(bands):
            row = i
            
            self.axes[band] = {}
            self.lines[band] = {}
            
            # Frequency plot
            self.axes[band]['freq'] = plt.subplot(5, 4, row*4 + 1)
            self.axes[band]['freq'].set_title(f'{band} Frequency', color=self.BAND_COLORS[band], fontsize=10)
            self.axes[band]['freq'].set_xlabel('Time (s)', fontsize=8)
            self.axes[band]['freq'].set_ylabel('Freq (Hz)', fontsize=8)
            self.axes[band]['freq'].grid(True, alpha=0.3)
            self.axes[band]['freq'].tick_params(labelsize=8)
            self.lines[band]['freq'], = self.axes[band]['freq'].plot([], [], 
                                                                   self.BAND_COLORS[band], linewidth=1.5)
            
            # Wavelength plot
            self.axes[band]['wave'] = plt.subplot(5, 4, row*4 + 2)
            self.axes[band]['wave'].set_title(f'{band} Wavelength', color=self.BAND_COLORS[band], fontsize=10)
            self.axes[band]['wave'].set_xlabel('Time (s)', fontsize=8)
            self.axes[band]['wave'].set_ylabel('λ (m)', fontsize=8)
            self.axes[band]['wave'].set_yscale('log')
            self.axes[band]['wave'].grid(True, alpha=0.3)
            self.axes[band]['wave'].tick_params(labelsize=8)
            self.lines[band]['wave'], = self.axes[band]['wave'].plot([], [], 
                                                                    self.BAND_COLORS[band], linewidth=1.5)
            
            # Amplitude plot
            self.axes[band]['amp'] = plt.subplot(5, 4, row*4 + 3)
            self.axes[band]['amp'].set_title(f'{band} Amplitude', color=self.BAND_COLORS[band], fontsize=10)
            self.axes[band]['amp'].set_xlabel('Time (s)', fontsize=8)
            self.axes[band]['amp'].set_ylabel('Amp', fontsize=8)
            self.axes[band]['amp'].grid(True, alpha=0.3)
            self.axes[band]['amp'].tick_params(labelsize=8)
            self.lines[band]['amp'], = self.axes[band]['amp'].plot([], [], 
                                                                  self.BAND_COLORS[band], linewidth=1.5)
            
            # Combined info plot
            self.axes[band]['info'] = plt.subplot(5, 4, row*4 + 4)
            self.axes[band]['info'].set_title(f'{band} Live Stats', color=self.BAND_COLORS[band], fontsize=10)
            self.axes[band]['info'].axis('off')
        
        plt.tight_layout()
        
    def setup_positioned_windows(self):
        """Setup separate windows with proper positioning - PREVENTS STACKING."""
        plt.style.use('dark_background')
        
        # Window positions - spread them across the screen
        positions = [
            (50, 50),      # Delta - top-left
            (450, 50),     # Theta - top-center  
            (850, 50),     # Alpha - top-right
            (50, 400),     # Beta - bottom-left
            (450, 400),    # Gamma - bottom-center
        ]
        
        bands = list(self.FREQUENCY_RANGES.keys())
        
        for i, band in enumerate(bands):
            # Create figure for this band
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f'{band} Brainwave Analysis - POSITIONED WINDOW', 
                        fontsize=14, color=self.BAND_COLORS[band])
            
            # CRITICAL: Position the window to prevent stacking
            mngr = fig.canvas.manager
            x, y = positions[i]
            mngr.window.wm_geometry(f"+{x}+{y}")
            
            self.figs[band] = fig
            
            # Create subplots for this band (2x2 grid)
            self.axes[band] = {}
            self.lines[band] = {}
            
            # Frequency plot
            self.axes[band]['freq'] = plt.subplot(2, 2, 1)
            self.axes[band]['freq'].set_title(f'{band} Frequency', color=self.BAND_COLORS[band])
            self.axes[band]['freq'].set_xlabel('Time (s)')
            self.axes[band]['freq'].set_ylabel('Frequency (Hz)')
            self.axes[band]['freq'].grid(True, alpha=0.3)
            self.lines[band]['freq'], = self.axes[band]['freq'].plot([], [], 
                                                                   self.BAND_COLORS[band], linewidth=2)
            
            # Wavelength plot
            self.axes[band]['wave'] = plt.subplot(2, 2, 2)
            self.axes[band]['wave'].set_title(f'{band} Wavelength', color=self.BAND_COLORS[band])
            self.axes[band]['wave'].set_xlabel('Time (s)')
            self.axes[band]['wave'].set_ylabel('Wavelength (m)')
            self.axes[band]['wave'].set_yscale('log')
            self.axes[band]['wave'].grid(True, alpha=0.3)
            self.lines[band]['wave'], = self.axes[band]['wave'].plot([], [], 
                                                                    self.BAND_COLORS[band], linewidth=2)
            
            # Amplitude plot
            self.axes[band]['amp'] = plt.subplot(2, 2, 3)
            self.axes[band]['amp'].set_title(f'{band} Amplitude', color=self.BAND_COLORS[band])
            self.axes[band]['amp'].set_xlabel('Time (s)')
            self.axes[band]['amp'].set_ylabel('Amplitude')
            self.axes[band]['amp'].grid(True, alpha=0.3)
            self.lines[band]['amp'], = self.axes[band]['amp'].plot([], [], 
                                                                  self.BAND_COLORS[band], linewidth=2)
            
            # Statistics display
            self.axes[band]['stats'] = plt.subplot(2, 2, 4)
            self.axes[band]['stats'].set_title(f'{band} Statistics', color=self.BAND_COLORS[band])
            self.axes[band]['stats'].axis('off')
            
            plt.tight_layout()
            
            # Force window to appear and update
            plt.show(block=False)
            plt.pause(0.1)
    
    def update_band_visualization(self, band, frame_num):
        """Update function for a specific band's visualization - OPTIMIZED."""
        self.update_count += 1
        
        # Get data safely
        times, freqs, waves, amps = self.get_band_data_safe(band)
        if times is None:
            return []
        
        try:
            # Update time series plots efficiently
            self.lines[band]['freq'].set_data(times, freqs)
            self.lines[band]['wave'].set_data(times, waves)
            self.lines[band]['amp'].set_data(times, amps)
            
            # Auto-scale axes for time series only when needed
            if frame_num % 10 == 0:  # Update scales every 10 frames
                for plot_type in ['freq', 'wave', 'amp']:
                    if plot_type in self.axes[band]:
                        self.axes[band][plot_type].relim()
                        self.axes[band][plot_type].autoscale_view()
            
            # Update statistics less frequently
            if frame_num % 20 == 0 and len(freqs) > 0:
                # Calculate current stats
                current_freq = freqs[-1]
                current_wave = waves[-1]
                current_amp = amps[-1]
                
                avg_freq = np.mean(freqs[-50:]) if len(freqs) >= 50 else np.mean(freqs)
                std_freq = np.std(freqs[-50:]) if len(freqs) >= 50 else np.std(freqs)
                avg_amp = np.mean(amps[-50:]) if len(amps) >= 50 else np.mean(amps)
                
                min_freq, max_freq = self.FREQUENCY_RANGES[band]
                
                # Performance stats
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    fps = self.update_count / elapsed if elapsed > 0 else 0
                else:
                    fps = 0
                
                if self.single_window_mode:
                    # Compact stats for single window
                    stats_text = f"""
{band}: {current_freq:.2f} Hz
λ: {current_wave:.2e} m
Amp: {current_amp:.3f}
Avg: {avg_freq:.2f}±{std_freq:.2f}
Samples: {len(freqs)}
FPS: {fps:.1f}
                    """
                    plot_key = 'info'
                else:
                    # Full stats for individual windows
                    stats_text = f"""
{band.upper()} BRAINWAVE ANALYSIS
{'═' * 40}
Current Values:
• Frequency: {current_freq:.6f} Hz
• Wavelength: {current_wave:.6e} m  
• Amplitude: {current_amp:.6f}

Statistics:
• Avg Frequency: {avg_freq:.3f} Hz
• Std Dev: {std_freq:.3f} Hz
• Avg Amplitude: {avg_amp:.3f}
• Range: {float(min_freq):.1f}-{float(max_freq):.1f} Hz
• Samples: {len(freqs)}

Performance:
• Update Rate: {fps:.1f} FPS
• Status: RUNNING
• Window: POSITIONED
                    """
                    plot_key = 'stats'
                
                if plot_key in self.axes[band]:
                    self.axes[band][plot_key].clear()
                    self.axes[band][plot_key].axis('off')
                    
                    font_size = 8 if self.single_window_mode else 10
                    self.axes[band][plot_key].text(0.02, 0.98, stats_text, 
                                                 transform=self.axes[band][plot_key].transAxes,
                                                 fontsize=font_size, fontfamily='monospace', 
                                                 color=self.BAND_COLORS[band],
                                                 verticalalignment='top')
            
            return [self.lines[band]['freq'], self.lines[band]['wave'], self.lines[band]['amp']]
            
        except Exception as e:
            print(f"Error updating {band}: {e}")
            return []
    
    def update_combined_visualization(self, frame_num):
        """Update function for single window mode."""
        artists = []
        for band in self.FREQUENCY_RANGES.keys():
            band_artists = self.update_band_visualization(band, frame_num)
            artists.extend(band_artists)
        return artists
    
    def start_live_analysis(self, duration_seconds=None):
        """Start real-time brainwave analysis - FIXED WINDOW MANAGEMENT."""
        print("Starting FIXED real-time brainwave analysis...")
        print(f"Mode: {'Single Window' if self.single_window_mode else 'Multiple Positioned Windows'}")
        print("Windows will be properly positioned to prevent stacking and freezing.")
        
        # Setup visualizations based on mode
        if self.single_window_mode:
            self.setup_single_window_visualization()
        else:
            self.setup_positioned_windows()
        
        # Start data generation thread
        self.is_running = True
        self.start_time = time.time()
        self.data_thread = threading.Thread(target=self.simulate_live_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Handle window close events
        def create_close_handler():
            def on_close(event):
                self.stop_live_analysis()
                plt.close('all')
            return on_close
        
        # CREATE ANIMATIONS
        if self.single_window_mode:
            # Single animation for combined window
            fig = self.figs['combined']
            fig.canvas.mpl_connect('close_event', create_close_handler())
            
            self.animations['combined'] = animation.FuncAnimation(
                fig,
                self.update_combined_visualization,
                interval=50,  # 20 FPS
                blit=False,
                cache_frame_data=False,
                repeat=True
            )
            print("Single window animation created")
        else:
            # Individual animations for each positioned window
            def create_update_function(band_name):
                def update_func(frame_num):
                    return self.update_band_visualization(band_name, frame_num)
                return update_func
            
            for band in self.FREQUENCY_RANGES.keys():
                fig = self.figs[band]
                fig.canvas.mpl_connect('close_event', create_close_handler())
                
                self.animations[band] = animation.FuncAnimation(
                    fig,
                    create_update_function(band),
                    interval=50,  # 20 FPS
                    blit=False,
                    cache_frame_data=False,
                    repeat=True
                )
                print(f"Animation created for {band} band at position")
        
        # Auto-stop after duration if specified
        if duration_seconds:
            def auto_stop():
                time.sleep(duration_seconds)
                self.stop_live_analysis()
                plt.close('all')
            
            auto_stop_thread = threading.Thread(target=auto_stop)
            auto_stop_thread.daemon = True
            auto_stop_thread.start()
        
        print(f"Started {len(self.animations)} animation(s)")
        print("You should now be able to move windows without freezing!")
        plt.show()
        return self.animations
    
    def stop_live_analysis(self):
        """Stop the real-time analysis."""
        print("Stopping real-time analysis...")
        self.is_running = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1)
    
    def get_live_summary(self):
        """Get summary of current live data for all bands."""
        summary = {}
        
        for band in self.FREQUENCY_RANGES.keys():
            times, freqs, waves, amps = self.get_band_data_safe(band)
            if freqs is None:
                summary[band] = "No data available"
                continue
            
            summary[band] = {
                'total_samples': len(freqs),
                'frequency_range': (min(freqs), max(freqs)),
                'current_frequency': freqs[-1],
                'current_wavelength': waves[-1],
                'current_amplitude': amps[-1],
                'average_frequency': np.mean(freqs),
                'average_amplitude': np.mean(amps)
            }
        
        return summary
    
    # Keep existing methods for backward compatibility
    def generate_frequency_range(self, min_freq: float, max_freq: float, num_frequencies: int = 5000) -> List[Decimal]:
        """Generate a range of frequencies with high precision."""
        if num_frequencies > self.max_frequencies:
            num_frequencies = self.max_frequencies
            warnings.warn(f"Limited to {self.max_frequencies} frequencies")
        
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        log_frequencies = np.linspace(log_min, log_max, num_frequencies)
        frequencies = [Decimal(str(round(10**log_freq, 11))) for log_freq in log_frequencies]
        
        return frequencies
    
    def calculate_high_precision_wavelength(self, frequency: Union[float, Decimal]) -> Decimal:
        """Calculate wavelength with high precision (11 decimal places)."""
        freq_decimal = Decimal(str(frequency))
        wavelength = self.speed_of_light / freq_decimal
        return wavelength.normalize()
    
    def classify_frequency(self, frequency: float) -> str:
        """Classify frequency into brainwave categories."""
        for wave_type, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
            if min_freq <= frequency < max_freq:
                return wave_type
        return 'Unknown'


def main():
    """FIXED VERSION with proper window management"""
    print("=" * 80)
    print("FIXED BRAINWAVE ANALYZER - PROPER WINDOW MANAGEMENT")
    print("No more stacking or freezing when moving windows!")
    print("=" * 80)
    
    # Choose display mode
    print("\nChoose display mode:")
    print("1. Single Window (all bands in one window) - RECOMMENDED")
    print("2. Multiple Windows (positioned separately)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    single_window = choice != '2'
    
    analyzer = BrainwaveAnalyzer(single_window_mode=single_window)
    
    print(f"\nMode: {'Single Window' if single_window else 'Multiple Positioned Windows'}")
    print("\nBrainwave bands:")
    for band, color in analyzer.BAND_COLORS.items():
        min_freq, max_freq = analyzer.FREQUENCY_RANGES[band]
        print(f"• {band}: {float(min_freq):.1f} - {float(max_freq):.1f} Hz")
    
    duration = input("\nEnter duration in seconds (or press Enter for indefinite): ").strip()
    duration = int(duration) if duration.isdigit() else None
    
    print(f"\nStarting FIXED real-time analysis...")
    print("FIXES APPLIED:")
    print("- Proper window positioning (no more stacking)")
    print("- Non-blocking event handling (no more freezing)")
    print("- Thread-safe data access")
    print("- Optimized animations")
    
    if single_window:
        print("- Single window mode prevents window management issues")
    else:
        print("- Multiple windows positioned at different screen locations")
    
    # Start live analysis
    animations = analyzer.start_live_analysis(duration_seconds=duration)
    
    # Show summary after stopping
    summary = analyzer.get_live_summary()
    print(f"\nAnalysis Summary:")
    for band, data in summary.items():
        if isinstance(data, dict):
            print(f"\n{band} Band:")
            print(f"  Samples: {data['total_samples']}")
            print(f"  Final frequency: {data['current_frequency']:.3f} Hz")
            print(f"  Average frequency: {data['average_frequency']:.3f} Hz")

if __name__ == "__main__":
    main()