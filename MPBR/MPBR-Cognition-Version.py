import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import warnings
from decimal import Decimal, getcontext
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
import json
from pathlib import Path
from collections import Counter

@dataclass
class ChakraData:
    """Data class for chakra information"""
    name: str
    frequency: float
    color: str
    element: str
    location: str
    properties: List[str]

class EnhancedBrainwaveAnalyzer:
    """
    Enhanced comprehensive brainwave and chakra analysis system for processing EEG-like data.
    Now includes advanced chakra analysis, quantum coherence metrics, and DNA resonance modeling.
    """
    
    def __init__(self):
        getcontext().prec = 18  # Increased precision
        self.speed_of_light = Decimal('299792458')
        self.max_frequencies = 10000  # Increased capacity
        
        # Enhanced chakra definitions with complete data
        self.CHAKRAS = {
            'Root': ChakraData('Root', 35.0, '#FF0000', 'Earth', 'Base of spine', 
                              ['Grounding', 'Survival', 'Stability', 'Security']),
            'Sacral': ChakraData('Sacral', 150.0, '#FF8800', 'Water', 'Lower abdomen', 
                               ['Creativity', 'Sexuality', 'Emotion', 'Pleasure']),
            'Solar_Plexus': ChakraData('Solar Plexus', 350.0, '#FFFF00', 'Fire', 'Upper abdomen', 
                                     ['Personal power', 'Confidence', 'Will', 'Transformation']),
            'Heart': ChakraData('Heart', 550.0, '#00FF00', 'Air', 'Center of chest', 
                              ['Love', 'Compassion', 'Connection', 'Healing']),
            'Throat': ChakraData('Throat', 750.0, '#0088FF', 'Sound', 'Throat', 
                               ['Communication', 'Truth', 'Expression', 'Clarity']),
            'Third_Eye': ChakraData('Third Eye', 950.0, '#4400FF', 'Light', 'Forehead', 
                                  ['Intuition', 'Wisdom', 'Vision', 'Insight']),
            'Crown': ChakraData('Crown', 1150.0, '#8800FF', 'Thought', 'Top of head', 
                              ['Spirituality', 'Connection to divine', 'Enlightenment', 'Unity'])
        }
        
        # Enhanced frequency ranges with more bands
        self.FREQUENCY_RANGES = {
            'Delta': (Decimal('0.5'), Decimal('4')),
            'Theta': (Decimal('4'), Decimal('8')),
            'Alpha': (Decimal('8'), Decimal('13')),
            'Beta': (Decimal('13'), Decimal('30')),
            'Gamma': (Decimal('30'), Decimal('100')),
            'High_Gamma': (Decimal('100'), Decimal('200')),
            'Ultra_Gamma': (Decimal('200'), Decimal('1000')),
            'Hyper_Gamma': (Decimal('1000'), Decimal('5000'))
        }
        
        # Enhanced associations
        self.CORTICAL_ASSOCIATIONS = {
            'Alpha': ['Occipital Lobe', 'Parietal Lobe'],
            'Beta': ['Frontal Lobe', 'Temporal Lobe'],
            'Theta': ['Temporal Lobe', 'Parietal Lobe', 'Hippocampus'],
            'Delta': ['Frontal Lobe', 'Occipital Lobe', 'Thalamus'],
            'Gamma': ['All Lobes', 'Thalamo-cortical circuits'],
            'High_Gamma': ['Prefrontal Cortex', 'Superior Temporal Gyrus'],
            'Ultra_Gamma': ['Thalamus', 'Brainstem', 'Cerebral Cortex'],
            'Hyper_Gamma': ['Quantum neural networks', 'Microtubules', 'Consciousness networks']
        }
        
        self.ACTIVITY_ASSOCIATIONS = {
            'Alpha': ['Relaxation', 'Reduced anxiety', 'Creativity', 'Wakeful rest'],
            'Beta': ['Alertness', 'Concentration', 'Problem-solving', 'Active thinking'],
            'Theta': ['Deep relaxation', 'Daydreaming', 'Meditation', 'Memory consolidation'],
            'Delta': ['Deep sleep', 'Unconsciousness', 'Healing', 'Regeneration'],
            'Gamma': ['Enhanced sensory processing', 'Information binding', 'Consciousness'],
            'High_Gamma': ['Advanced cognitive processing', 'Heightened awareness'],
            'Ultra_Gamma': ['Hyper-synchronization', 'Global neural binding', 'Peak cognitive performance'],
            'Hyper_Gamma': ['Transcendent consciousness', 'Unity experiences', 'Non-local awareness']
        }
    
    def calculate_quantum_coherence(self, frequency: float, amplitude: float = 1.0) -> Dict:
        """Calculate quantum coherence metrics for given frequency"""
        planck_constant = 6.62607015e-34
        energy = planck_constant * frequency
        
        # Quantum coherence metrics
        coherence_factor = np.exp(-energy / (1.381e-23 * 310))  # At body temperature
        decoherence_time = 1 / (2 * np.pi * frequency)
        quantum_phase = (2 * np.pi * frequency * amplitude) % (2 * np.pi)
        
        return {
            'energy_joules': energy,
            'coherence_factor': coherence_factor,
            'decoherence_time_seconds': decoherence_time,
            'quantum_phase': quantum_phase,
            'quantum_number': int(energy / planck_constant),
            'temperature_kelvin': energy / 1.381e-23 if energy > 0 else 0
        }
    
    def calculate_dna_resonance(self, frequency: float, dna_freq: float = 1e8) -> Dict:
        """Enhanced DNA resonance calculation with multiple harmonics"""
        base_freq = frequency
        harmonics = [base_freq * (i + 1) for i in range(10)]
        
        # Calculate resonance with DNA frequency
        resonance_ratios = [dna_freq / harmonic for harmonic in harmonics]
        closest_harmonic_idx = np.argmin([abs(ratio - round(ratio)) for ratio in resonance_ratios])
        
        # Helix interaction modeling
        dna_wavelength = 3.4e-9  # meters (one base pair)
        electromagnetic_wavelength = self.speed_of_light / Decimal(str(frequency))
        
        # Calculate interference patterns
        interference_ratio = float(electromagnetic_wavelength) / dna_wavelength
        standing_wave_nodes = int(interference_ratio / 2)
        
        return {
            'base_frequency': base_freq,
            'harmonics': harmonics,
            'resonance_ratios': resonance_ratios,
            'optimal_harmonic': harmonics[closest_harmonic_idx],
            'dna_wavelength_meters': dna_wavelength,
            'em_wavelength_meters': float(electromagnetic_wavelength),
            'interference_ratio': interference_ratio,
            'standing_wave_nodes': standing_wave_nodes,
            'resonance_strength': 1 / (1 + abs(resonance_ratios[closest_harmonic_idx] - round(resonance_ratios[closest_harmonic_idx])))
        }
    
    def analyze_chakra_system(self, frequencies: Optional[List[float]] = None) -> Dict:
        """Comprehensive chakra system analysis"""
        if frequencies is None:
            frequencies = [chakra.frequency for chakra in self.CHAKRAS.values()]
        
        chakra_analyses = {}
        system_metrics = {
            'total_energy': 0,
            'coherence_sum': 0,
            'balance_score': 0,
            'dominant_chakra': None,
            'energy_distribution': {}
        }
        
        for i, (chakra_name, chakra_data) in enumerate(self.CHAKRAS.items()):
            freq = frequencies[i] if i < len(frequencies) else chakra_data.frequency
            
            # Individual chakra analysis
            quantum_metrics = self.calculate_quantum_coherence(freq)
            dna_resonance = self.calculate_dna_resonance(freq)
            brainwave_metrics = self.calculate_wavelength_metrics({'pulseFrequency': freq})
            
            chakra_analysis = {
                'chakra_data': chakra_data,
                'frequency': freq,
                'quantum_metrics': quantum_metrics,
                'dna_resonance': dna_resonance,
                'brainwave_metrics': brainwave_metrics,
                'energy_level': quantum_metrics['energy_joules'],
                'activation_score': quantum_metrics['coherence_factor'] * dna_resonance['resonance_strength']
            }
            
            chakra_analyses[chakra_name] = chakra_analysis
            
            # Update system metrics
            system_metrics['total_energy'] += quantum_metrics['energy_joules']
            system_metrics['coherence_sum'] += quantum_metrics['coherence_factor']
            system_metrics['energy_distribution'][chakra_name] = quantum_metrics['energy_joules']
        
        # Calculate system balance
        energies = list(system_metrics['energy_distribution'].values())
        system_metrics['balance_score'] = 1 - (np.std(energies) / np.mean(energies)) if np.mean(energies) > 0 else 0
        system_metrics['dominant_chakra'] = max(system_metrics['energy_distribution'], 
                                               key=system_metrics['energy_distribution'].get)
        
        return {
            'individual_chakras': chakra_analyses,
            'system_metrics': system_metrics,
            'recommendations': self._generate_chakra_recommendations(chakra_analyses)
        }
    
    def _generate_chakra_recommendations(self, chakra_analyses: Dict) -> List[str]:
        """Generate personalized recommendations based on chakra analysis"""
        recommendations = []
        
        for chakra_name, analysis in chakra_analyses.items():
            activation = analysis['activation_score']
            chakra_data = analysis['chakra_data']
            
            if activation < 0.3:
                recommendations.append(f"Consider {chakra_data.element.lower()} element practices for {chakra_name} chakra")
                recommendations.append(f"Focus on {chakra_data.properties[0].lower()} exercises")
            elif activation > 0.8:
                recommendations.append(f"{chakra_name} chakra is highly active - maintain balance")
        
        return recommendations
    
    def create_advanced_visualization(self, comprehensive_results: Dict, chakra_results: Dict = None, 
                                    save_path: str = None):
        """Create advanced visualization with chakra and quantum data"""
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(24, 16))
        
        # Main frequency analysis (top row)
        results = comprehensive_results['analysis_results']
        frequencies = np.array(results['frequencies'])
        wavelengths = np.array(results['wavelengths'])
        
        # 1. Frequency-Wavelength relationship
        ax1 = plt.subplot(3, 4, 1)
        scatter = ax1.scatter(frequencies, wavelengths, c=frequencies, cmap='plasma', s=10, alpha=0.7)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Wavelength (m)')
        ax1.set_title('Frequency vs Wavelength')
        plt.colorbar(scatter, ax=ax1, label='Frequency (Hz)')
        
        # 2. Brainwave band distribution
        ax2 = plt.subplot(3, 4, 2)
        band_counts = comprehensive_results['summary_statistics']['band_distribution']
        colors = plt.cm.viridis(np.linspace(0, 1, len(band_counts)))
        wedges, texts, autotexts = ax2.pie(band_counts.values(), labels=band_counts.keys(), 
                                          autopct='%1.1f%%', colors=colors)
        ax2.set_title('Brainwave Band Distribution')
        
        # 3. Frequency spectrum
        ax3 = plt.subplot(3, 4, 3)
        ax3.hist(frequencies, bins=50, alpha=0.7, color='cyan', edgecolor='white', linewidth=0.5)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Count')
        ax3.set_title('Frequency Spectrum')
        ax3.set_yscale('log')
        
        # 4. Wavelength distribution
        ax4 = plt.subplot(3, 4, 4)
        ax4.hist(wavelengths, bins=50, alpha=0.7, color='orange', edgecolor='white', linewidth=0.5)
        ax4.set_xlabel('Wavelength (m)')
        ax4.set_ylabel('Count')
        ax4.set_title('Wavelength Distribution')
        ax4.set_xscale('log')
        
        # Chakra visualization (middle row)
        if chakra_results:
            # 5. Chakra energy levels
            ax5 = plt.subplot(3, 4, 5)
            chakra_names = list(chakra_results['individual_chakras'].keys())
            energies = [chakra_results['individual_chakras'][name]['energy_level'] 
                       for name in chakra_names]
            colors = [chakra_results['individual_chakras'][name]['chakra_data'].color 
                     for name in chakra_names]
            
            bars = ax5.bar(range(len(chakra_names)), energies, color=colors, alpha=0.8)
            ax5.set_xlabel('Chakras')
            ax5.set_ylabel('Energy Level (J)')
            ax5.set_title('Chakra Energy Distribution')
            ax5.set_xticks(range(len(chakra_names)))
            ax5.set_xticklabels([name.replace('_', ' ') for name in chakra_names], rotation=45)
            ax5.set_yscale('log')
            
            # 6. Chakra activation scores
            ax6 = plt.subplot(3, 4, 6)
            activation_scores = [chakra_results['individual_chakras'][name]['activation_score'] 
                               for name in chakra_names]
            ax6.plot(range(len(chakra_names)), activation_scores, 'o-', linewidth=2, markersize=8, color='gold')
            ax6.set_xlabel('Chakras')
            ax6.set_ylabel('Activation Score')
            ax6.set_title('Chakra Activation Profile')
            ax6.set_xticks(range(len(chakra_names)))
            ax6.set_xticklabels([name.replace('_', ' ') for name in chakra_names], rotation=45)
            ax6.grid(True, alpha=0.3)
            
            # 7. Chakra balance radar chart
            ax7 = plt.subplot(3, 4, 7, projection='polar')
            angles = np.linspace(0, 2*np.pi, len(chakra_names), endpoint=False)
            activation_scores_normalized = np.array(activation_scores) / max(activation_scores)
            
            ax7.plot(angles, activation_scores_normalized, 'o-', linewidth=2, color='cyan')
            ax7.fill(angles, activation_scores_normalized, alpha=0.25, color='cyan')
            ax7.set_xticks(angles)
            ax7.set_xticklabels([name.replace('_', ' ') for name in chakra_names])
            ax7.set_title('Chakra Balance Radar')
            ax7.set_ylim(0, 1)
            
            # 8. DNA resonance strength
            ax8 = plt.subplot(3, 4, 8)
            resonance_strengths = [chakra_results['individual_chakras'][name]['dna_resonance']['resonance_strength'] 
                                 for name in chakra_names]
            ax8.scatter(range(len(chakra_names)), resonance_strengths, 
                       c=colors, s=100, alpha=0.8, edgecolors='white', linewidth=2)
            ax8.set_xlabel('Chakras')
            ax8.set_ylabel('DNA Resonance Strength')
            ax8.set_title('DNA-Chakra Resonance')
            ax8.set_xticks(range(len(chakra_names)))
            ax8.set_xticklabels([name.replace('_', ' ') for name in chakra_names], rotation=45)
            ax8.grid(True, alpha=0.3)
        
        # Quantum and advanced metrics (bottom row)
        # 9. Phase space plot
        ax9 = plt.subplot(3, 4, 9)
        sample_size = min(1000, len(frequencies))
        freq_sample = frequencies[:sample_size]
        wave_sample = wavelengths[:sample_size]
        
        # Create phase space representation
        phase_x = freq_sample * np.cos(2 * np.pi * freq_sample / max(freq_sample))
        phase_y = wave_sample * np.sin(2 * np.pi * wave_sample / max(wave_sample))
        
        ax9.scatter(phase_x, phase_y, c=freq_sample, cmap='plasma', s=20, alpha=0.6)
        ax9.set_xlabel('Phase X')
        ax9.set_ylabel('Phase Y')
        ax9.set_title('Frequency Phase Space')
        
        # 10. Spectral density
        ax10 = plt.subplot(3, 4, 10)
        freq_bins = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 50)
        density, _ = np.histogram(frequencies, bins=freq_bins)
        bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
        
        ax10.loglog(bin_centers, density + 1, 'b-', linewidth=2)
        ax10.fill_between(bin_centers, density + 1, alpha=0.3)
        ax10.set_xlabel('Frequency (Hz)')
        ax10.set_ylabel('Spectral Density')
        ax10.set_title('Power Spectral Density')
        ax10.grid(True, alpha=0.3)
        
        # 11. Statistics summary
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        stats = comprehensive_results['summary_statistics']
        stats_text = f"""
        COMPREHENSIVE ANALYSIS SUMMARY
        {'='*35}
        Total Frequencies: {stats['total_analyzed']:,}
        Precision: 15+ decimal places
        
        Frequency Statistics:
        • Range: {stats['frequency_stats']['min']:.2f} - {stats['frequency_stats']['max']:.2f} Hz
        • Mean: {stats['frequency_stats']['mean']:.2f} Hz
        • Std Dev: {stats['frequency_stats']['std']:.2f} Hz
        
        Wavelength Statistics:
        • Range: {stats['wavelength_stats']['min']:.2e} - {stats['wavelength_stats']['max']:.2e} m
        • Mean: {stats['wavelength_stats']['mean']:.2e} m
        
        Band Distribution:
        """
        
        for band, percentage in stats['band_percentages'].items():
            if percentage > 0:
                stats_text += f"• {band}: {percentage:.1f}%\n        "
        
        if chakra_results:
            stats_text += f"""
        
        Chakra System:
        • Balance Score: {chakra_results['system_metrics']['balance_score']:.3f}
        • Dominant: {chakra_results['system_metrics']['dominant_chakra'].replace('_', ' ')}
        • Total Energy: {chakra_results['system_metrics']['total_energy']:.2e} J
        """
        
        ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, fontsize=9, 
                 verticalalignment='top', fontfamily='monospace', color='white')
        
        # 12. 3D frequency landscape
        ax12 = plt.subplot(3, 4, 12, projection='3d')
        
        # Create 3D surface of frequency distribution
        x_edges = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 20)
        y_edges = np.logspace(np.log10(min(wavelengths)), np.log10(max(wavelengths)), 20)
        
        H, xedges, yedges = np.histogram2d(frequencies, wavelengths, bins=[x_edges, y_edges])
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        
        surface = ax12.plot_surface(np.log10(X), np.log10(Y), H.T, cmap='plasma', alpha=0.8)
        ax12.set_xlabel('log10(Frequency)')
        ax12.set_ylabel('log10(Wavelength)')
        ax12.set_zlabel('Density')
        ax12.set_title('3D Frequency Landscape')
        
        plt.suptitle('Enhanced Brainwave & Chakra Analysis Dashboard', fontsize=20, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        plt.show()
    
    def generate_comprehensive_frequency_analysis(self, min_freq: float = 0.5, max_freq: float = 2000.0, 
                                                num_frequencies: int = 5000) -> Dict:
        """Generate comprehensive frequency analysis across specified range"""
        # Generate frequency range
        frequencies = self.generate_frequency_range(min_freq, max_freq, num_frequencies)
        
        # Calculate metrics for each frequency
        results = {
            'frequencies': [],
            'wavelengths': [],
            'metrics': [],
            'quantum_data': [],
            'dna_resonance': []
        }
        
        for freq in frequencies:
            freq_float = float(freq)
            
            # Basic wavelength calculation
            wavelength = self.calculate_high_precision_wavelength(freq)
            
            # Comprehensive metrics
            metrics = self.calculate_wavelength_metrics({'pulseFrequency': freq_float})
            quantum_data = self.calculate_quantum_coherence(freq_float)
            dna_data = self.calculate_dna_resonance(freq_float)
            
            results['frequencies'].append(freq_float)
            results['wavelengths'].append(float(wavelength))
            results['metrics'].append(metrics)
            results['quantum_data'].append(quantum_data)
            results['dna_resonance'].append(dna_data)
        
        # Calculate summary statistics
        frequencies_array = np.array(results['frequencies'])
        wavelengths_array = np.array(results['wavelengths'])
        
        # Band distribution
        band_counts = {}
        for freq in frequencies_array:
            band = self.classify_frequency(freq)
            band_counts[band] = band_counts.get(band, 0) + 1
        
        summary_stats = {
            'total_analyzed': len(frequencies_array),
            'frequency_stats': {
                'min': float(np.min(frequencies_array)),
                'max': float(np.max(frequencies_array)),
                'mean': float(np.mean(frequencies_array)),
                'std': float(np.std(frequencies_array))
            },
            'wavelength_stats': {
                'min': float(np.min(wavelengths_array)),
                'max': float(np.max(wavelengths_array)),
                'mean': float(np.mean(wavelengths_array)),
                'std': float(np.std(wavelengths_array))
            },
            'band_distribution': band_counts,
            'band_percentages': {band: (count/len(frequencies_array))*100 
                               for band, count in band_counts.items()}
        }
        
        return {
            'analysis_results': results,
            'summary_statistics': summary_stats,
            'parameters': {
                'min_frequency': min_freq,
                'max_frequency': max_freq,
                'num_frequencies': num_frequencies
            }
        }

    def export_comprehensive_data(self, comprehensive_results: Dict, chakra_results: Dict = None, 
                                filename: str = "enhanced_analysis.json"):
        """Export all analysis data to JSON format"""
        export_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'frequency_analysis': comprehensive_results,
            'metadata': {
                'analyzer_version': '2.0',
                'precision_decimal_places': getcontext().prec,
                'max_frequencies': self.max_frequencies
            }
        }
        
        if chakra_results:
            # Convert chakra data to serializable format
            serializable_chakra_results = {}
            for key, value in chakra_results.items():
                if key == 'individual_chakras':
                    serializable_chakra_results[key] = {}
                    for chakra_name, chakra_data in value.items():
                        serialized_chakra = dict(chakra_data)
                        # Convert ChakraData to dict
                        if 'chakra_data' in serialized_chakra:
                            chakra_obj = serialized_chakra['chakra_data']
                            serialized_chakra['chakra_data'] = {
                                'name': chakra_obj.name,
                                'frequency': chakra_obj.frequency,
                                'color': chakra_obj.color,
                                'element': chakra_obj.element,
                                'location': chakra_obj.location,
                                'properties': chakra_obj.properties
                            }
                        serializable_chakra_results[key][chakra_name] = serialized_chakra
                else:
                    serializable_chakra_results[key] = value
            
            export_data['chakra_analysis'] = serializable_chakra_results
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Comprehensive data exported to {filename}")
    
    # Inherit all original methods with enhancements
    def generate_frequency_range(self, min_freq: float, max_freq: float, num_frequencies: int = 10000) -> List[Decimal]:
        """Enhanced frequency range generation with multiple distribution options"""
        if num_frequencies > self.max_frequencies:
            num_frequencies = self.max_frequencies
            warnings.warn(f"Limited to {self.max_frequencies} frequencies")
        
        # Use both linear and logarithmic distribution for better coverage
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        
        # 70% logarithmic, 30% linear for better distribution
        num_log = int(0.7 * num_frequencies)
        num_lin = num_frequencies - num_log
        
        log_frequencies = np.logspace(log_min, log_max, num_log)
        lin_frequencies = np.linspace(min_freq, max_freq, num_lin)
        
        all_frequencies = np.concatenate([log_frequencies, lin_frequencies])
        all_frequencies = np.unique(np.sort(all_frequencies))[:num_frequencies]
        
        frequencies = [Decimal(str(round(freq, 11))) for freq in all_frequencies]
        return frequencies
    
    def calculate_high_precision_wavelength(self, frequency: Union[float, Decimal]) -> Decimal:
        """Enhanced wavelength calculation with error handling"""
        try:
            freq_decimal = Decimal(str(frequency))
            if freq_decimal <= 0:
                raise ValueError("Frequency must be positive")
            wavelength = self.speed_of_light / freq_decimal
            return wavelength.normalize()
        except Exception as e:
            warnings.warn(f"Wavelength calculation error: {e}")
            return Decimal('0')
    
    # Keep all original methods for backward compatibility
    def calculate_wavelength_metrics(self, data: Dict) -> Dict:
        """Enhanced wavelength metrics calculation"""
        self.validate_data(data)
        
        pulse_amplitude = data.get('pulseAmplitude', 0.0001)
        pulse_frequency = Decimal(str(data['pulseFrequency']))
        magnetic_field_direction = data.get('magneticFieldDirection', 1.0)
        
        # Core calculations
        wavelength = self.speed_of_light / pulse_frequency
        wavelength_class = self.classify_frequency(float(pulse_frequency))
        
        # Enhanced metrics with quantum calculations
        quantum_metrics = self.calculate_quantum_coherence(float(pulse_frequency), pulse_amplitude)
        dna_metrics = self.calculate_dna_resonance(float(pulse_frequency))
        
        # Original categorization
        if wavelength <= Decimal('100.0'):
            wavelength_category = "High Frequency (Low Wavelength)"
        elif wavelength <= Decimal('1000.0'):
            wavelength_category = "Medium Frequency (Medium Wavelength)"
        else:
            wavelength_category = "Low Frequency (High Wavelength)"
        
        if pulse_amplitude < 0.3:
            wavelength_pattern = "Low Amplitude (Stable)"
        elif pulse_amplitude < 0.7:
            wavelength_pattern = "Medium Amplitude (Variable)"
        else:
            wavelength_pattern = "High Amplitude (Dynamic)"
        
        # Enhanced metrics
        metrics = {
            'pulse_amplitude': pulse_amplitude,
            'pulse_frequency': float(pulse_frequency),
            'magnetic_field_direction': magnetic_field_direction,
            'wavelength': float(wavelength),
            'wavelength_class': wavelength_class,
            'wavelength_category': wavelength_category,
            'wavelength_pattern': wavelength_pattern,
            'frequency_range': self.FREQUENCY_RANGES.get(wavelength_class, (0, 0)),
            'cortical_regions': self.CORTICAL_ASSOCIATIONS.get(wavelength_class, ['Unknown']),
            'associated_activities': self.ACTIVITY_ASSOCIATIONS.get(wavelength_class, ['Unknown']),
            'power_estimate': pulse_amplitude ** 2,
            'energy_density': (pulse_amplitude ** 2) / float(wavelength) if wavelength != 0 else 0,
            'quantum_metrics': quantum_metrics,
            'dna_resonance': dna_metrics
        }
        
        return metrics
    
    def validate_data(self, data: Dict) -> bool:
        """Enhanced data validation"""
        required_keys = ['pulseFrequency']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(data[key], (int, float)) or data[key] <= 0:
                raise ValueError(f"Invalid value for {key}: must be positive number")
        
        # Additional validations
        if data['pulseFrequency'] > 1e12:
            warnings.warn("Extremely high frequency detected - results may be unrealistic")
        
        return True
    
    def classify_frequency(self, frequency: float) -> str:
        """Enhanced frequency classification"""
        for wave_type, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
            if min_freq <= frequency < max_freq:
                return wave_type
        return 'Unknown'

# Enhanced analysis functions for each chakra
def enhanced_chakra_analysis(chakra_name: str, analyzer: EnhancedBrainwaveAnalyzer) -> Dict:
    """Perform enhanced analysis for a specific chakra"""
    chakra_data = analyzer.CHAKRAS[chakra_name]
    
    # Multiple frequency analysis around the base frequency
    base_freq = chakra_data.frequency
    frequencies = [base_freq * (0.9 + 0.02 * i) for i in range(10)]  # ±10% variation
    
    results = []
    for freq in frequencies:
        metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
        quantum_metrics = analyzer.calculate_quantum_coherence(freq)
        dna_resonance = analyzer.calculate_dna_resonance(freq)
        
        results.append({
            'frequency': freq,
            'metrics': metrics,
            'quantum': quantum_metrics,
            'dna': dna_resonance
        })
    
    # Find optimal frequency
    optimal_result = max(results, key=lambda x: x['dna']['resonance_strength'])
    
    # Create analysis structure compatible with _generate_chakra_recommendations
    analysis_for_recommendations = {
        'chakra_data': chakra_data,
        'activation_score': optimal_result['quantum']['coherence_factor'] * optimal_result['dna']['resonance_strength']
    }
    
    return {
        'chakra_data': chakra_data,
        'base_frequency': base_freq,
        'optimal_frequency': optimal_result['frequency'],
        'frequency_range_analysis': results,
        'optimal_metrics': optimal_result,
        'recommendations': analyzer._generate_chakra_recommendations({chakra_name: analysis_for_recommendations})
    }

if __name__ == '__main__':
    # Create enhanced analyzer
    analyzer = EnhancedBrainwaveAnalyzer()
    
    print("=" * 80)
    print("ENHANCED BRAINWAVE & CHAKRA ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Generate comprehensive frequency analysis
    comprehensive_results = analyzer.generate_comprehensive_frequency_analysis(
        min_freq=0.5, max_freq=2000.0, num_frequencies=5000
    )
    
    # Analyze chakra system
    chakra_results = analyzer.analyze_chakra_system()
    
    # Create advanced visualization
    analyzer.create_advanced_visualization(comprehensive_results, chakra_results, 
                                         save_path="enhanced_analysis.png")
    
    # Export comprehensive data
    analyzer.export_comprehensive_data(comprehensive_results, chakra_results)
    
    # Individual chakra analyses
    print("\nINDIVIDUAL CHAKRA ANALYSES:")
    print("-" * 50)
    
    for chakra_name in analyzer.CHAKRAS.keys():
        enhanced_analysis = enhanced_chakra_analysis(chakra_name, analyzer)
        print(f"\n{chakra_name.replace('_', ' ')} Chakra:")
        print(f"  Base Frequency: {enhanced_analysis['base_frequency']:.1f} Hz")
        print(f"  Optimal Frequency: {enhanced_analysis['optimal_frequency']:.2f} Hz")
        print(f"  Element: {enhanced_analysis['chakra_data'].element}")
        print(f"  Properties: {', '.join(enhanced_analysis['chakra_data'].properties[:2])}")
        
        optimal = enhanced_analysis['optimal_metrics']
        print(f"  DNA Resonance: {optimal['dna']['resonance_strength']:.4f}")
        print(f"  Quantum Coherence: {optimal['quantum']['coherence_factor']:.6f}")
    
    # System summary
    print(f"\nSYSTEM SUMMARY:")
    print(f"  Total Chakra Energy: {chakra_results['system_metrics']['total_energy']:.2e} J")
    print(f"  System Balance Score: {chakra_results['system_metrics']['balance_score']:.3f}")
    print(f"  Dominant Chakra: {chakra_results['system_metrics']['dominant_chakra'].replace('_', ' ')}")
    
    print("\nAnalysis complete! Check 'enhanced_analysis.png' and 'enhanced_analysis.json'")


#-----------------------------------------------------------------------------------------------------------

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
        'Delta': (Decimal('0.5'), Decimal('4')),
        'Theta': (Decimal('4'), Decimal('8')),
        'Alpha': (Decimal('8'), Decimal('13')),
        'Beta' : (Decimal('13'), Decimal('30')),
        'Gamma': (Decimal('30'), Decimal('100')),
        # add an ultra-high band
        'UltraGamma': (Decimal('100'), Decimal('2000'))
    }
    CORTICAL_ASSOCIATIONS = {
        'Alpha': ['Occipital Lobe', 'Parietal Lobe'],
        'Beta':  ['Frontal Lobe', 'Temporal Lobe'],
        'Theta': ['Temporal Lobe', 'Parietal Lobe'],
        'Delta': ['Frontal Lobe', 'Occipital Lobe'],
        'Gamma': ['All Lobes'],
        'UltraGamma': ['Thalamus', 'Brainstem', 'Cerebral Cortex']
    }
    
    ACTIVITY_ASSOCIATIONS = {
        'Alpha':      ['Relaxation', 'Reduced anxiety', 'Creativity', 'Wakeful rest'],
        'Beta':       ['Alertness', 'Concentration', 'Problem-solving', 'Active thinking'],
        'Theta':      ['Deep relaxation', 'Daydreaming', 'Meditation', 'Memory consolidation'],
        'Delta':      ['Deep sleep', 'Unconsciousness', 'Healing', 'Regeneration'],
        'Gamma':      ['Enhanced sensory processing', 'Information binding', 'Consciousness'],
        'UltraGamma': ['Hyper-synchronization', 'Global neural binding', 'Peak cognitive performance']
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
        };
        
        return summary;
    
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

        # Convert frequency to Decimal before dividing
        pulse_frequency = Decimal(str(data['pulseFrequency']))

        magnetic_field_direction = data.get('magneticFieldDirection', 1.0)
        
        # Core calculations
        wavelength = self.speed_of_light / pulse_frequency
        wavelength_class = self.classify_frequency(float(pulse_frequency))
        
        # Wavelength categorization
        if wavelength <= Decimal('100.0'):
            wavelength_category = "High Frequency (Low Wavelength)"
        elif wavelength <= Decimal('1000.0'):
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
            'pulse_frequency': float(pulse_frequency),
            'magnetic_field_direction': magnetic_field_direction,
            'wavelength': float(wavelength),
            'wavelength_class': wavelength_class,
            'wavelength_category': wavelength_category,
            'wavelength_pattern': wavelength_pattern,
            'frequency_range': self.FREQUENCY_RANGES.get(wavelength_class, (0, 0)),
            'cortical_regions': self.CORTICAL_ASSOCIATIONS.get(wavelength_class, ['Unknown']),
            'associated_activities': self.ACTIVITY_ASSOCIATIONS.get(wavelength_class, ['Unknown']),
            'power_estimate': pulse_amplitude ** 2,
            'energy_density': (pulse_amplitude ** 2) / float(wavelength) if wavelength != 0 else 0
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
        
        # Handle dominant_frequency safely
        psd_half = psd[:len(psd)//2]
        if psd_half.size > 0:
            dominant_idx = np.argmax(psd_half)
            dominant_frequency = freq_array[dominant_idx] if dominant_idx < len(freq_array) else float('nan')
        else:
            dominant_frequency = float('nan')

        features = {
            'mean_frequency': np.mean(freq_array),
            'frequency_std': np.std(freq_array),
            'dominant_frequency': dominant_frequency,
            'spectral_centroid': np.sum(freq_array * psd[:len(freq_array)]) / np.sum(psd[:len(freq_array)]) if np.sum(psd[:len(freq_array)]) != 0 else float('nan'),
            'spectral_bandwidth': np.sqrt(np.sum(((freq_array - np.mean(freq_array)) ** 2) * psd[:len(freq_array)]) / np.sum(psd[:len(freq_array)])) if np.sum(psd[:len(freq_array)]) != 0 else float('nan'),
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
print("-------------------------------------------------------------------------------------------------")

import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_crown_chakra():
    """Analyze crown chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 480.0  # Crown chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Crown Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_crown_chakra()


import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_heart_chakra():
    """Analyze heart chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 341.3  # Heart chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Heart Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_heart_chakra()


import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_root_chakra():
    """Analyze root chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 256.0  # Root chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Root Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_root_chakra()


import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_sacral_chakra():
    """Analyze sacral chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 288.0  # Typical sacral chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Sacral Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    # Simple quantum demonstration with cupy-derived angle
    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_sacral_chakra()


import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_solar_plexus_chakra():
    """Analyze solar plexus chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 320.0  # Solar Plexus chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Solar Plexus Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_solar_plexus_chakra()


import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_third_eye_chakra():
    """Analyze third eye chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 426.7  # Third Eye chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Third Eye Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_third_eye_chakra()


import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def analyze_throat_chakra():
    """Analyze throat chakra interaction with psyche using MPBR."""
    analyzer = BrainwaveAnalyzer()
    freq = 384.0  # Throat chakra frequency in Hz

    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})

    print("Throat Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)

    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Quantum Statevector:", state)


if __name__ == "__main__":
    analyze_throat_chakra()

import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def run_quantum_simulation(shots: int = 128) -> dict:
    """Perform a simple quantum circuit simulation."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    return result.get_counts()


def analyze_chakra(name: str, min_freq: float, max_freq: float, num_points: int = 128) -> dict:
    """Analyze a chakra-related frequency range using BrainwaveAnalyzer."""
    analyzer = BrainwaveAnalyzer()
    results = analyzer.generate_comprehensive_frequency_analysis(
        min_freq=min_freq,
        max_freq=max_freq,
        num_frequencies=num_points
    )
    frequencies = cp.asarray(results['analysis_results']['frequencies'])
    mean_freq_gpu = float(cp.mean(frequencies).get())

    return {
        'chakra': name,
        'mean_frequency': mean_freq_gpu,
        'quantum_counts': run_quantum_simulation(),
        'results': results
    }

if __name__ == "__main__":
    info = analyze_chakra("Crown Chakra", 40.0, 60.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])


if __name__ == "__main__":
    info = analyze_chakra("Heart Chakra", 13.0, 20.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])


if __name__ == "__main__":
    info = analyze_chakra("Root Chakra", 1.0, 4.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])

if __name__ == "__main__":
    info = analyze_chakra("Sacral Chakra", 4.0, 8.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])



if __name__ == "__main__":
    info = analyze_chakra("Solar Plexus Chakra", 8.0, 13.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])


if __name__ == "__main__":
    info = analyze_chakra("Third Eye Chakra", 30.0, 40.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])



if __name__ == "__main__":
    info = analyze_chakra("Throat Chakra", 20.0, 30.0)
    print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
    print("Quantum counts:", info['quantum_counts'])

import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def quantum_chakra_rotation(freq: float) -> dict:
    """Simulate a quantum rotation for a chakra frequency and return statevector."""
    angle = float(cp.pi / 4)
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    return {
        'angle': angle,
        'statevector': state
    }

def analyze_single_chakra(chakra_name: str, freq: float):
    """Analyze a single chakra's psychological and quantum metrics."""
    analyzer = BrainwaveAnalyzer()
    metrics = analyzer.calculate_wavelength_metrics({'pulseFrequency': freq})
    analysis = analyzer.analyze_brainwave_state({'pulseFrequency': freq})
    quantum = quantum_chakra_rotation(freq)
    print(f"{chakra_name} Chakra Psychological Metrics:")
    print(metrics)
    print("Brainwave State Analysis:")
    print(analysis)
    print("Quantum Statevector:", quantum['statevector'])

def run_quantum_simulation(shots: int = 128) -> dict:
    """Perform a simple quantum circuit simulation."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    return result.get_counts()

def analyze_chakra_range(name: str, min_freq: float, max_freq: float, num_points: int = 128) -> dict:
    """Analyze a chakra-related frequency range using BrainwaveAnalyzer and GPU acceleration."""
    analyzer = BrainwaveAnalyzer()
    results = analyzer.generate_comprehensive_frequency_analysis(
        min_freq=min_freq,
        max_freq=max_freq,
        num_frequencies=num_points
    )
    frequencies = cp.asarray(results['analysis_results']['frequencies'])
    mean_freq_gpu = float(cp.mean(frequencies).get())
    return {
        'chakra': name,
        'mean_frequency': mean_freq_gpu,
        'quantum_counts': run_quantum_simulation(),
        'results': results
    }

if __name__ == "__main__":
    # Analyze individual chakras with psychological and quantum metrics
    chakra_freqs = [
        ("Crown", 480.0),
        ("Heart", 341.3),
        ("Root", 256.0),
        ("Sacral", 288.0),
        ("Solar Plexus", 320.0),
        ("Third Eye", 426.7),
        ("Throat", 384.0)
    ]
    for name, freq in chakra_freqs:
        analyze_single_chakra(name, freq)
        print("-" * 60)

    # Analyze chakra frequency ranges with quantum simulation and GPU mean
    chakra_ranges = [
        ("Crown Chakra", 40.0, 60.0),
        ("Heart Chakra", 13.0, 20.0),
        ("Root Chakra", 1.0, 4.0),
        ("Sacral Chakra", 4.0, 8.0),
        ("Solar Plexus Chakra", 8.0, 13.0),
        ("Third Eye Chakra", 30.0, 40.0),
        ("Throat Chakra", 20.0, 30.0)
    ]
    for name, min_f, max_f in chakra_ranges:
        info = analyze_chakra_range(name, min_f, max_f)
        print(f"{info['chakra']} mean frequency: {info['mean_frequency']:.4f} Hz")
        print("Quantum counts:", info['quantum_counts'])
        print("=" * 60)
