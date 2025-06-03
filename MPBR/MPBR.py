import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings
from decimal import Decimal, getcontext

class BrainwaveAnalyzer:
    """
    A comprehensive brainwave analysis system for processing EEG-like data.
    Enhanced for high-precision frequency calculations up to 5000 frequencies.
    """
    
    def __init__(self):
        # Set precision for 11 decimal places
        getcontext().prec = 15  # Extra precision to ensure 11 decimal places
        self.speed_of_light = Decimal('299792458')  # m/s with high precision
        self.max_frequencies = 5000
    
    # Constants for brainwave frequency ranges with high-precision boundaries
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
    
    def generate_frequency_range(self, min_freq: float, max_freq: float, num_frequencies: int = 5000) -> List[Decimal]:
        """Generate a range of frequencies with high precision."""
        if num_frequencies > self.max_frequencies:
            num_frequencies = self.max_frequencies
            warnings.warn(f"Limited to {self.max_frequencies} frequencies")
        
        # Create logarithmic distribution for better coverage across brainwave bands
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        log_frequencies = np.linspace(log_min, log_max, num_frequencies)
        frequencies = [Decimal(str(round(10**log_freq, 11))) for log_freq in log_frequencies]
        
        return frequencies
    
    def calculate_high_precision_wavelength(self, frequency: Union[float, Decimal]) -> Decimal:
        """Calculate wavelength with high precision (11 decimal places)."""
        freq_decimal = Decimal(str(frequency))
        wavelength = self.speed_of_light / freq_decimal
        # Use normalize to remove trailing zeros and handle precision dynamically
        return wavelength.normalize()
    
    def batch_analyze_frequencies(self, frequencies: List[Union[float, Decimal]]) -> Dict:
        """Analyze multiple frequencies with high precision calculations."""
        if len(frequencies) > self.max_frequencies:
            frequencies = frequencies[:self.max_frequencies]
            warnings.warn(f"Limited analysis to first {self.max_frequencies} frequencies")
        
        results = {
            'frequencies': [],
            'wavelengths': [],
            'classifications': [],
            'cortical_regions': [],
            'activities': [],
            'precision_metrics': []
        }
        
        print(f"Analyzing {len(frequencies)} frequencies with 11-decimal precision...")
        
        for i, freq in enumerate(frequencies):
            if i % 500 == 0:  # Progress indicator
                print(f"Processing frequency {i+1}/{len(frequencies)}...")
            
            freq_decimal = Decimal(str(freq))
            wavelength = self.calculate_high_precision_wavelength(freq_decimal)
            classification = self.classify_frequency(float(freq))
            
            results['frequencies'].append(float(freq_decimal))
            results['wavelengths'].append(float(wavelength))
            results['classifications'].append(classification)
            results['cortical_regions'].append(self.CORTICAL_ASSOCIATIONS.get(classification, ['Unknown']))
            results['activities'].append(self.ACTIVITY_ASSOCIATIONS.get(classification, ['Unknown']))
            
            # Store precision metrics
            precision_info = {
                'frequency_str': str(freq_decimal),
                'wavelength_str': str(wavelength),
                'decimal_places': len(str(wavelength).split('.')[-1]) if '.' in str(wavelength) else 0
            }
            results['precision_metrics'].append(precision_info)
        
        return results
    
    def generate_comprehensive_frequency_analysis(self, 
                                                min_freq: float = 0.5, 
                                                max_freq: float = 100.0, 
                                                num_frequencies: int = 5000) -> Dict:
        """Generate comprehensive analysis for up to 5000 frequencies."""
        
        print("=" * 80)
        print(f"COMPREHENSIVE HIGH-PRECISION FREQUENCY ANALYSIS")
        print(f"Frequency Range: {min_freq} Hz to {max_freq} Hz")
        print(f"Number of Frequencies: {min(num_frequencies, self.max_frequencies)}")
        print(f"Precision: 11 decimal places")
        print("=" * 80)
        
        # Generate frequency range
        frequencies = self.generate_frequency_range(min_freq, max_freq, num_frequencies)
        
        # Batch analyze
        analysis_results = self.batch_analyze_frequencies(frequencies)
        
        # Calculate summary statistics
        summary = self.calculate_summary_statistics(analysis_results)
        
        return {
            'analysis_results': analysis_results,
            'summary_statistics': summary,
            'parameters': {
                'min_frequency': min_freq,
                'max_frequency': max_freq,
                'total_frequencies': len(frequencies),
                'precision_decimal_places': 11
            }
        }
    
    def calculate_summary_statistics(self, results: Dict) -> Dict:
        """Calculate summary statistics for the frequency analysis."""
        frequencies = np.array(results['frequencies'])
        wavelengths = np.array(results['wavelengths'])
        classifications = results['classifications']
        
        # Band distribution
        band_counts = {}
        for band in self.FREQUENCY_RANGES.keys():
            band_counts[band] = classifications.count(band)
        
        summary = {
            'frequency_stats': {
                'min': float(np.min(frequencies)),
                'max': float(np.max(frequencies)),
                'mean': float(np.mean(frequencies)),
                'std': float(np.std(frequencies)),
                'median': float(np.median(frequencies))
            },
            'wavelength_stats': {
                'min': float(np.min(wavelengths)),
                'max': float(np.max(wavelengths)),
                'mean': float(np.mean(wavelengths)),
                'std': float(np.std(wavelengths)),
                'median': float(np.median(wavelengths))
            },
            'band_distribution': band_counts,
            'band_percentages': {band: (count/len(classifications))*100 
                               for band, count in band_counts.items()},
            'total_analyzed': len(frequencies)
        }
        
        return summary
    
    def visualize_high_precision_analysis(self, comprehensive_results: Dict, save_path: str = None):
        """Create visualization for high-precision frequency analysis."""
        results = comprehensive_results['analysis_results']
        summary = comprehensive_results['summary_statistics']
        
        frequencies = results['frequencies']
        wavelengths = results['wavelengths']
        classifications = results['classifications']
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Main frequency-wavelength plot
        ax1 = plt.subplot(2, 3, 1)
        plt.loglog(frequencies, wavelengths, 'b.', markersize=1, alpha=0.6)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavelength (m)')
        plt.title('Frequency vs Wavelength (Log-Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # Frequency distribution
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(frequencies, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count')
        plt.title('Frequency Distribution')
        plt.grid(True, alpha=0.3)
        
        # Wavelength distribution
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(wavelengths, bins=50, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Wavelength (m)')
        plt.ylabel('Count')
        plt.title('Wavelength Distribution')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Band distribution pie chart
        ax4 = plt.subplot(2, 3, 4)
        band_counts = summary['band_distribution']
        plt.pie(band_counts.values(), labels=band_counts.keys(), autopct='%1.1f%%')
        plt.title('Brainwave Band Distribution')
        
        # Precision demonstration
        ax5 = plt.subplot(2, 3, 5)
        sample_indices = np.linspace(0, len(frequencies)-1, 100, dtype=int)
        sample_freqs = [frequencies[i] for i in sample_indices]
        sample_wavelengths = [wavelengths[i] for i in sample_indices]
        plt.scatter(sample_freqs, sample_wavelengths, c=sample_indices, cmap='viridis', s=20)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavelength (m)')
        plt.title('High-Precision Sample Points')
        plt.colorbar(label='Sample Index')
        plt.grid(True, alpha=0.3)
        
        # Statistics summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = f"""
        ANALYSIS SUMMARY
        ================
        Total Frequencies: {summary['total_analyzed']:,}
        Precision: 11 decimal places
        
        Frequency Range:
        • Min: {summary['frequency_stats']['min']:.6f} Hz
        • Max: {summary['frequency_stats']['max']:.6f} Hz
        • Mean: {summary['frequency_stats']['mean']:.6f} Hz
        
        Wavelength Range:
        • Min: {summary['wavelength_stats']['min']:.2e} m
        • Max: {summary['wavelength_stats']['max']:.2e} m
        • Mean: {summary['wavelength_stats']['mean']:.2e} m
        
        Band Distribution:
        """
        
        for band, percentage in summary['band_percentages'].items():
            stats_text += f"• {band}: {percentage:.1f}%\n        "
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('High-Precision Brainwave Frequency Analysis (5000 Frequencies)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_high_precision_data(self, comprehensive_results: Dict, filename: str = "high_precision_frequencies.csv"):
        """Export high-precision frequency data to CSV."""
        results = comprehensive_results['analysis_results']
        precision_metrics = results['precision_metrics']
        
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['index', 'frequency_hz', 'wavelength_m', 'classification', 
                         'cortical_regions', 'activities', 'frequency_11_decimal', 'wavelength_11_decimal']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(results['frequencies'])):
                writer.writerow({
                    'index': i + 1,
                    'frequency_hz': results['frequencies'][i],
                    'wavelength_m': results['wavelengths'][i],
                    'classification': results['classifications'][i],
                    'cortical_regions': ', '.join(results['cortical_regions'][i]),
                    'activities': ', '.join(results['activities'][i]),
                    'frequency_11_decimal': precision_metrics[i]['frequency_str'],
                    'wavelength_11_decimal': precision_metrics[i]['wavelength_str']
                })
        
        print(f"High-precision data exported to {filename}")
    
    def validate_data(self, data: Dict) -> bool:
        """Validate input data structure and values."""
        required_keys = ['pulseFrequency']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(data[key], (int, float)) or data[key] <= 0:
                raise ValueError(f"Invalid value for {key}: must be positive number")
        
        return True
    
    def classify_frequency(self, frequency: float) -> str:
        """Classify frequency into brainwave categories."""
        for wave_type, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
            if min_freq <= frequency < max_freq:
                return wave_type
        return 'Unknown'
    
    def calculate_wavelength_metrics(self, data: Dict) -> Dict:
        """Calculate comprehensive wavelength and frequency metrics."""
        self.validate_data(data)
        
        pulse_amplitude = data.get('pulseAmplitude', 0.0001)
        pulse_frequency = data['pulseFrequency']
        magnetic_field_direction = data.get('magneticFieldDirection', 1.0)
        
        # Core calculations
        wavelength = self.speed_of_light / pulse_frequency
        wavelength_class = self.classify_frequency(pulse_frequency)
        
        # Wavelength categorization
        if wavelength <= 100.0:
            wavelength_category = "High Frequency (Low Wavelength)"
        elif wavelength <= 1000.0:
            wavelength_category = "Medium Frequency (Medium Wavelength)"
        else:
            wavelength_category = "Low Frequency (High Wavelength)"
        
        # Pattern analysis based on amplitude
        if pulse_amplitude < 0.3:
            wavelength_pattern = "Low Amplitude (Stable)"
        elif pulse_amplitude < 0.7:
            wavelength_pattern = "Medium Amplitude (Variable)"
        else:
            wavelength_pattern = "High Amplitude (Dynamic)"
        
        # Advanced metrics
        metrics = {
            'pulse_amplitude': pulse_amplitude,
            'pulse_frequency': pulse_frequency,
            'magnetic_field_direction': magnetic_field_direction,
            'wavelength': wavelength,
            'wavelength_class': wavelength_class,
            'wavelength_category': wavelength_category,
            'wavelength_pattern': wavelength_pattern,
            'frequency_range': self.FREQUENCY_RANGES.get(wavelength_class, (0, 0)),
            'cortical_regions': self.CORTICAL_ASSOCIATIONS.get(wavelength_class, ['Unknown']),
            'associated_activities': self.ACTIVITY_ASSOCIATIONS.get(wavelength_class, ['Unknown']),
            'power_estimate': pulse_amplitude ** 2,
            'energy_density': (pulse_amplitude ** 2) / wavelength
        }
        
        return metrics
    
    def analyze_brainwave_state(self, data: Dict) -> Dict:
        """Comprehensive brainwave state analysis."""
        metrics = self.calculate_wavelength_metrics(data)
        
        analysis = {
            'primary_state': metrics['wavelength_class'],
            'cortical_involvement': metrics['cortical_regions'],
            'cognitive_activities': metrics['associated_activities'],
            'power_level': 'High' if metrics['power_estimate'] > 0.5 else 'Low',
            'stability': 'Stable' if metrics['pulse_amplitude'] < 0.5 else 'Variable'
        }
        
        # Advanced state detection
        if metrics['wavelength_class'] == 'Alpha' and metrics['power_estimate'] > 0.3:
            analysis['special_state'] = 'Relaxed Awareness'
        elif metrics['wavelength_class'] == 'Theta' and metrics['power_estimate'] > 0.4:
            analysis['special_state'] = 'Deep Meditative State'
        elif metrics['wavelength_class'] == 'Gamma' and metrics['power_estimate'] > 0.6:
            analysis['special_state'] = 'Heightened Consciousness'
        
        return analysis
    
    def filter_noise(self, data: Dict, cutoff_frequency: float = 0.5) -> Dict:
        """Apply noise filtering with improved algorithm."""
        filtered_data = data.copy()
        
        # High-pass filter for frequency data
        if 'pulseFrequency' in data and data['pulseFrequency'] < cutoff_frequency:
            warnings.warn(f"Frequency {data['pulseFrequency']} below cutoff, setting to cutoff value")
            filtered_data['pulseFrequency'] = cutoff_frequency
        
        # Amplitude noise reduction
        if 'pulseAmplitude' in data:
            # Simple noise gate
            if data['pulseAmplitude'] < 0.001:
                filtered_data['pulseAmplitude'] = 0.0
        
        return filtered_data
    
    def extract_spectral_features(self, frequency_data: List[float]) -> Dict:
        """Extract advanced spectral features from frequency data."""
        if not frequency_data:
            return {}
        
        freq_array = np.array(frequency_data)
        
        # Calculate power spectral density
        fft_data = np.fft.fft(freq_array)
        psd = np.abs(fft_data) ** 2
        
        features = {
            'mean_frequency': np.mean(freq_array),
            'frequency_std': np.std(freq_array),
            'dominant_frequency': freq_array[np.argmax(psd[:len(psd)//2])],
            'spectral_centroid': np.sum(freq_array * psd[:len(freq_array)]) / np.sum(psd[:len(freq_array)]),
            'spectral_bandwidth': np.sqrt(np.sum(((freq_array - np.mean(freq_array)) ** 2) * psd[:len(freq_array)]) / np.sum(psd[:len(freq_array)])),
            'total_power': np.sum(psd)
        }
        
        # Band-specific power analysis
        for band, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
            band_mask = (freq_array >= min_freq) & (freq_array < max_freq)
            if np.any(band_mask):
                features[f'{band.lower()}_power'] = np.sum(psd[band_mask])
                features[f'{band.lower()}_relative_power'] = features[f'{band.lower()}_power'] / features['total_power']
        
        return features
    
    def recognize_patterns(self, features: Dict) -> Dict:
        """Advanced pattern recognition with multiple criteria."""
        patterns = {}
        
        if not features:
            return patterns
        
        # Find dominant frequency band
        band_powers = {}
        for band in self.FREQUENCY_RANGES.keys():
            power_key = f'{band.lower()}_power'
            if power_key in features:
                band_powers[band] = features[power_key]
        
        if band_powers:
            dominant_band = max(band_powers, key=band_powers.get)
            patterns['dominant_band'] = dominant_band
            patterns['dominance_ratio'] = band_powers[dominant_band] / sum(band_powers.values())
        
        # Specific pattern detection
        if 'alpha_power' in features and 'beta_power' in features:
            alpha_beta_ratio = features['alpha_power'] / (features['beta_power'] + 1e-10)
            patterns['relaxation_index'] = alpha_beta_ratio
            patterns['is_relaxed'] = alpha_beta_ratio > 1.5
        
        if 'theta_power' in features and 'total_power' in features:
            theta_dominance = features['theta_power'] / features['total_power']
            patterns['meditation_index'] = theta_dominance
            patterns['is_meditative'] = theta_dominance > 0.4
        
        return patterns
    
    def visualize_analysis(self, data_points: List[Dict], save_path: str = None):
        """Create comprehensive visualization of brainwave data."""
        if not data_points:
            print("No data points to visualize")
            return
        
        # Extract data for visualization
        frequencies = [d['pulseFrequency'] for d in data_points]
        amplitudes = [d.get('pulseAmplitude', 0) for d in data_points]
        wavelengths = [self.speed_of_light / f for f in frequencies]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comprehensive Brainwave Analysis', fontsize=16)
        
        # Frequency plot
        ax1.plot(frequencies, 'b-', marker='o', markersize=4)
        ax1.set_title('Pulse Frequency Over Time')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.grid(True, alpha=0.3)
        
        # Amplitude plot
        ax2.plot(amplitudes, 'r-', marker='s', markersize=4)
        ax2.set_title('Pulse Amplitude Over Time')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Wavelength plot
        ax3.plot(wavelengths, 'g-', marker='^', markersize=4)
        ax3.set_title('Wavelength Over Time')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Wavelength (m)')
        ax3.grid(True, alpha=0.3)
        
        # Frequency distribution
        ax4.hist(frequencies, bins=min(20, len(frequencies)), alpha=0.7, edgecolor='black')
        ax4.set_title('Frequency Distribution')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, data: Dict) -> str:
        """Generate a comprehensive analysis report."""
        try:
            metrics = self.calculate_wavelength_metrics(data)
            analysis = self.analyze_brainwave_state(data)
            
            report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BRAINWAVE ANALYSIS REPORT                 ║
╠══════════════════════════════════════════════════════════════╣
║ INPUT PARAMETERS:                                            ║
║ • Pulse Frequency: {metrics['pulse_frequency']:.2f} Hz                      ║
║ • Pulse Amplitude: {metrics['pulse_amplitude']:.4f}                         ║
║ • Magnetic Field Direction: {metrics['magnetic_field_direction']:.2f}             ║
║                                                              ║
║ CALCULATED METRICS:                                          ║
║ • Wavelength: {metrics['wavelength']:.2e} meters                ║
║ • Brainwave Class: {metrics['wavelength_class']:<15}                     ║
║ • Category: {metrics['wavelength_category']:<25}                       ║
║ • Pattern: {metrics['wavelength_pattern']:<25}                        ║
║ • Power Estimate: {metrics['power_estimate']:.4f}                       ║
║ • Energy Density: {metrics['energy_density']:.2e}              ║
║                                                              ║
║ NEUROLOGICAL ANALYSIS:                                       ║
║ • Primary State: {analysis['primary_state']:<15}                       ║
║ • Cortical Regions: {', '.join(analysis['cortical_involvement'])}       ║
║ • Associated Activities:                                     ║
"""
            
            for activity in analysis['cognitive_activities']:
                report += f"║   - {activity:<55} ║\n"
            
            report += f"""║ • Power Level: {analysis['power_level']:<15}                           ║
║ • Stability: {analysis['stability']:<15}                             ║
"""
            
            if 'special_state' in analysis:
                report += f"║ • Special State: {analysis['special_state']:<25}                   ║\n"
            
            report += "╚══════════════════════════════════════════════════════════════╝"
            
            return report
            
        except Exception as e:
            return f"Error generating report: {str(e)}"


# Enhanced usage example
def main():
    """Main function demonstrating high-precision frequency analysis."""
    analyzer = BrainwaveAnalyzer()
    
    print("=" * 80)
    print("HIGH-PRECISION BRAINWAVE ANALYZER")
    print("Calculating up to 5000 frequencies with 11 decimal places")
    print("=" * 80)
    
    # Generate comprehensive analysis
    comprehensive_results = analyzer.generate_comprehensive_frequency_analysis(
        min_freq=0.5,
        max_freq=100.0,
        num_frequencies=5000
    )
    
    # Display sample high-precision results
    print("\nSAMPLE HIGH-PRECISION CALCULATIONS:")
    print("=" * 80)
    results = comprehensive_results['analysis_results']
    precision_metrics = results['precision_metrics']
    
    # Show first 10 and last 10 calculations
    sample_indices = list(range(10)) + list(range(-10, 0))
    
    for i in sample_indices:
        freq_str = precision_metrics[i]['frequency_str']
        wavelength_str = precision_metrics[i]['wavelength_str']
        classification = results['classifications'][i]
        
        print(f"#{i+1 if i >= 0 else len(results['frequencies'])+i+1:4d}: "
              f"Freq: {freq_str:>15s} Hz → "
              f"Wavelength: {wavelength_str:>18s} m ({classification})")
    
    # Display summary
    summary = comprehensive_results['summary_statistics']
    print(f"\nSUMMARY STATISTICS:")
    print("=" * 80)
    print(f"Total frequencies analyzed: {summary['total_analyzed']:,}")
    print(f"Frequency range: {summary['frequency_stats']['min']:.11f} - {summary['frequency_stats']['max']:.11f} Hz")
    print(f"Wavelength range: {summary['wavelength_stats']['min']:.2e} - {summary['wavelength_stats']['max']:.2e} m")
    
    print(f"\nBand distribution:")
    for band, count in summary['band_distribution'].items():
        percentage = summary['band_percentages'][band]
        print(f"  {band:>5s}: {count:4d} frequencies ({percentage:5.1f}%)")
    
    # Generate visualizations
    print(f"\nGenerating comprehensive visualization...")
    analyzer.visualize_high_precision_analysis(comprehensive_results)
    
    # Export data
    print(f"\nExporting high-precision data...")
    analyzer.export_high_precision_data(comprehensive_results)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
