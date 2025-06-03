import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings
from decimal import Decimal, getcontext
import json
from datetime import datetime
import logging

class BrainwaveAnalyzer:
    """
    A comprehensive brainwave analysis system for processing EEG-like data.
    Enhanced for high-precision frequency calculations up to 5000 frequencies.
    Now includes z-score algorithms for pattern tracking and logging.
    """
    
    def __init__(self):
        # Set precision for 11 decimal places
        getcontext().prec = 15  # Extra precision to ensure 11 decimal places
        self.speed_of_light = Decimal('299792458')  # m/s with high precision
        self.max_frequencies = 5000
        
        # Z-score tracking initialization
        self.baseline_stats = {}
        self.pattern_history = []
        self.z_score_threshold = 2.0  # Standard threshold for anomaly detection
        self.pattern_log = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for brainwave pattern tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('brainwave_patterns.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainwaveAnalyzer')
    
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
    
    def establish_baseline(self, frequency_data: List[float], amplitude_data: List[float] = None) -> Dict:
        """Establish baseline statistics for z-score calculations."""
        if not frequency_data:
            raise ValueError("Frequency data cannot be empty for baseline establishment")
        
        freq_array = np.array(frequency_data)
        
        baseline = {
            'frequency': {
                'mean': np.mean(freq_array),
                'std': np.std(freq_array),
                'count': len(freq_array),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if amplitude_data:
            amp_array = np.array(amplitude_data)
            baseline['amplitude'] = {
                'mean': np.mean(amp_array),
                'std': np.std(amp_array),
                'count': len(amp_array),
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate band-specific baselines
        for band, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
            band_mask = (freq_array >= float(min_freq)) & (freq_array < float(max_freq))
            if np.any(band_mask):
                band_freqs = freq_array[band_mask]
                baseline[f'{band.lower()}_band'] = {
                    'mean': np.mean(band_freqs),
                    'std': np.std(band_freqs),
                    'count': len(band_freqs),
                    'percentage': (len(band_freqs) / len(freq_array)) * 100
                }
        
        self.baseline_stats = baseline
        self.logger.info(f"Baseline established with {len(frequency_data)} frequency samples")
        
        return baseline
    
    def calculate_z_scores(self, current_data: Dict) -> Dict:
        """Calculate z-scores for current data against baseline."""
        if not self.baseline_stats:
            raise ValueError("Baseline must be established before calculating z-scores")
        
        z_scores = {
            'timestamp': datetime.now().isoformat(),
            'raw_data': current_data.copy()
        }
        
        # Calculate frequency z-score
        if 'pulseFrequency' in current_data and 'frequency' in self.baseline_stats:
            baseline_freq = self.baseline_stats['frequency']
            if baseline_freq['std'] > 0:
                freq_z = (current_data['pulseFrequency'] - baseline_freq['mean']) / baseline_freq['std']
                z_scores['frequency_z_score'] = freq_z
                z_scores['frequency_anomaly'] = abs(freq_z) > self.z_score_threshold
            else:
                z_scores['frequency_z_score'] = 0.0
                z_scores['frequency_anomaly'] = False
        
        # Calculate amplitude z-score
        if 'pulseAmplitude' in current_data and 'amplitude' in self.baseline_stats:
            baseline_amp = self.baseline_stats['amplitude']
            if baseline_amp['std'] > 0:
                amp_z = (current_data['pulseAmplitude'] - baseline_amp['mean']) / baseline_amp['std']
                z_scores['amplitude_z_score'] = amp_z
                z_scores['amplitude_anomaly'] = abs(amp_z) > self.z_score_threshold
            else:
                z_scores['amplitude_z_score'] = 0.0
                z_scores['amplitude_anomaly'] = False
        
        # Calculate band-specific z-scores
        current_freq = current_data.get('pulseFrequency', 0)
        current_band = self.classify_frequency(current_freq)
        
        if current_band != 'Unknown':
            band_key = f'{current_band.lower()}_band'
            if band_key in self.baseline_stats:
                baseline_band = self.baseline_stats[band_key]
                if baseline_band['std'] > 0:
                    band_z = (current_freq - baseline_band['mean']) / baseline_band['std']
                    z_scores[f'{current_band.lower()}_band_z_score'] = band_z
                    z_scores[f'{current_band.lower()}_band_anomaly'] = abs(band_z) > self.z_score_threshold
        
        # Overall anomaly assessment
        anomaly_indicators = [
            z_scores.get('frequency_anomaly', False),
            z_scores.get('amplitude_anomaly', False)
        ]
        z_scores['overall_anomaly'] = any(anomaly_indicators)
        z_scores['anomaly_count'] = sum(anomaly_indicators)
        
        return z_scores
    
    def track_pattern_changes(self, z_scores: Dict) -> Dict:
        """Track and analyze pattern changes using z-scores."""
        pattern_analysis = {
            'timestamp': z_scores['timestamp'],
            'pattern_type': 'normal',
            'severity': 'low',
            'description': 'Normal brainwave activity within baseline parameters'
        }
        
        # Determine pattern type based on z-scores
        if z_scores.get('overall_anomaly', False):
            freq_z = z_scores.get('frequency_z_score', 0)
            amp_z = z_scores.get('amplitude_z_score', 0)
            
            # Classify pattern types
            if abs(freq_z) > 3.0:
                pattern_analysis['pattern_type'] = 'extreme_frequency_deviation'
                pattern_analysis['severity'] = 'high'
                pattern_analysis['description'] = f"Extreme frequency deviation (z-score: {freq_z:.2f})"
            elif abs(amp_z) > 3.0:
                pattern_analysis['pattern_type'] = 'extreme_amplitude_deviation'
                pattern_analysis['severity'] = 'high'
                pattern_analysis['description'] = f"Extreme amplitude deviation (z-score: {amp_z:.2f})"
            elif freq_z > 2.0 and amp_z > 2.0:
                pattern_analysis['pattern_type'] = 'high_activity_state'
                pattern_analysis['severity'] = 'medium'
                pattern_analysis['description'] = "Both frequency and amplitude elevated"
            elif freq_z < -2.0 and amp_z < -2.0:
                pattern_analysis['pattern_type'] = 'low_activity_state'
                pattern_analysis['severity'] = 'medium'
                pattern_analysis['description'] = "Both frequency and amplitude suppressed"
            else:
                pattern_analysis['pattern_type'] = 'mild_anomaly'
                pattern_analysis['severity'] = 'low'
                pattern_analysis['description'] = "Mild deviation from baseline"
        
        # Add to pattern history
        self.pattern_history.append(pattern_analysis)
        
        # Keep only last 1000 patterns to manage memory
        if len(self.pattern_history) > 1000:
            self.pattern_history = self.pattern_history[-1000:]
        
        return pattern_analysis
    
    def log_brainwave_pattern(self, data: Dict, z_scores: Dict, pattern_analysis: Dict):
        """Log brainwave patterns with z-score analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_data': data,
            'z_scores': z_scores,
            'pattern_analysis': pattern_analysis,
            'baseline_age_minutes': self._calculate_baseline_age()
        }
        
        # Add to pattern log
        self.pattern_log.append(log_entry)
        
        # Log significant patterns
        if pattern_analysis['severity'] in ['medium', 'high']:
            self.logger.warning(
                f"Significant pattern detected: {pattern_analysis['pattern_type']} "
                f"(Severity: {pattern_analysis['severity']}) - {pattern_analysis['description']}"
            )
        elif z_scores.get('overall_anomaly', False):
            self.logger.info(
                f"Anomaly detected: {pattern_analysis['description']} "
                f"(Freq Z: {z_scores.get('frequency_z_score', 0):.2f}, "
                f"Amp Z: {z_scores.get('amplitude_z_score', 0):.2f})"
            )
        
        # Keep log manageable
        if len(self.pattern_log) > 5000:
            self.pattern_log = self.pattern_log[-5000:]
    
    def _calculate_baseline_age(self) -> float:
        """Calculate age of baseline in minutes."""
        if not self.baseline_stats or 'frequency' not in self.baseline_stats:
            return 0.0
        
        baseline_time = datetime.fromisoformat(self.baseline_stats['frequency']['timestamp'])
        current_time = datetime.now()
        age_seconds = (current_time - baseline_time).total_seconds()
        return age_seconds / 60.0
    
    def detect_pattern_clusters(self, lookback_minutes: int = 60) -> Dict:
        """Detect clusters of similar patterns within a time window."""
        if not self.pattern_history:
            return {'clusters': [], 'summary': 'No pattern history available'}
        
        # Filter patterns within lookback window
        cutoff_time = datetime.now().timestamp() - (lookback_minutes * 60)
        recent_patterns = []
        
        for pattern in self.pattern_history:
            pattern_time = datetime.fromisoformat(pattern['timestamp']).timestamp()
            if pattern_time >= cutoff_time:
                recent_patterns.append(pattern)
        
        if not recent_patterns:
            return {'clusters': [], 'summary': f'No patterns in last {lookback_minutes} minutes'}
        
        # Group patterns by type
        pattern_groups = {}
        for pattern in recent_patterns:
            pattern_type = pattern['pattern_type']
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)
        
        # Identify significant clusters
        clusters = []
        for pattern_type, patterns in pattern_groups.items():
            if len(patterns) >= 3:  # Minimum cluster size
                cluster = {
                    'pattern_type': pattern_type,
                    'count': len(patterns),
                    'severity_distribution': {},
                    'first_occurrence': patterns[0]['timestamp'],
                    'last_occurrence': patterns[-1]['timestamp'],
                    'duration_minutes': (
                        datetime.fromisoformat(patterns[-1]['timestamp']).timestamp() -
                        datetime.fromisoformat(patterns[0]['timestamp']).timestamp()
                    ) / 60.0
                }
                
                # Calculate severity distribution
                for pattern in patterns:
                    severity = pattern['severity']
                    cluster['severity_distribution'][severity] = (
                        cluster['severity_distribution'].get(severity, 0) + 1
                    )
                
                clusters.append(cluster)
        
        # Sort clusters by significance (count * max severity weight)
        severity_weights = {'low': 1, 'medium': 2, 'high': 3}
        for cluster in clusters:
            max_severity = max(cluster['severity_distribution'].keys(), 
                             key=lambda x: severity_weights.get(x, 0))
            cluster['significance_score'] = (
                cluster['count'] * severity_weights.get(max_severity, 1)
            )
        
        clusters.sort(key=lambda x: x['significance_score'], reverse=True)
        
        return {
            'clusters': clusters,
            'summary': f'Found {len(clusters)} pattern clusters in last {lookback_minutes} minutes',
            'total_patterns': len(recent_patterns),
            'lookback_minutes': lookback_minutes
        }
    
    def generate_z_score_report(self, data: Dict) -> str:
        """Generate comprehensive z-score analysis report."""
        if not self.baseline_stats:
            return "No baseline established. Please run establish_baseline() first."
        
        try:
            # Calculate current metrics
            z_scores = self.calculate_z_scores(data)
            pattern_analysis = self.track_pattern_changes(z_scores)
            self.log_brainwave_pattern(data, z_scores, pattern_analysis)
            
            # Get recent clusters
            clusters = self.detect_pattern_clusters(60)
            
            report = f"""
╔══════════════════════════════════════════════════════════════╗
║                  Z-SCORE BRAINWAVE ANALYSIS                  ║
╠══════════════════════════════════════════════════════════════╣
║ CURRENT MEASUREMENT:                                         ║
║ • Frequency: {data.get('pulseFrequency', 0):.2f} Hz                             ║
║ • Amplitude: {data.get('pulseAmplitude', 0):.4f}                            ║
║ • Timestamp: {z_scores['timestamp'][:19]}                    ║
║                                                              ║
║ Z-SCORE ANALYSIS:                                            ║
║ • Frequency Z-Score: {z_scores.get('frequency_z_score', 0):>8.2f}                      ║
║ • Amplitude Z-Score: {z_scores.get('amplitude_z_score', 0):>8.2f}                      ║
║ • Overall Anomaly: {str(z_scores.get('overall_anomaly', False)):>10s}                    ║
║ • Anomaly Count: {z_scores.get('anomaly_count', 0):>12d}                        ║
║                                                              ║
║ PATTERN ANALYSIS:                                            ║
║ • Pattern Type: {pattern_analysis['pattern_type']:<25}              ║
║ • Severity: {pattern_analysis['severity']:<15}                           ║
║ • Description: {pattern_analysis['description'][:35]:<35}    ║
║                                                              ║
║ BASELINE INFORMATION:                                        ║
║ • Baseline Age: {self._calculate_baseline_age():>8.1f} minutes                       ║
║ • Frequency Mean: {self.baseline_stats['frequency']['mean']:>10.2f} Hz                   ║
║ • Frequency Std: {self.baseline_stats['frequency']['std']:>11.2f} Hz                   ║
"""
            
            if 'amplitude' in self.baseline_stats:
                report += f"║ • Amplitude Mean: {self.baseline_stats['amplitude']['mean']:>10.4f}                     ║\n"
                report += f"║ • Amplitude Std: {self.baseline_stats['amplitude']['std']:>11.4f}                     ║\n"
            
            report += f"""║                                                              ║
║ RECENT PATTERN CLUSTERS ({clusters['lookback_minutes']} min window):              ║
║ • Total Patterns: {clusters['total_patterns']:>12d}                        ║
║ • Clusters Found: {len(clusters['clusters']):>12d}                         ║
"""
            
            # Add top 3 clusters
            for i, cluster in enumerate(clusters['clusters'][:3]):
                report += f"║ • Cluster {i+1}: {cluster['pattern_type'][:20]:<20} (Count: {cluster['count']:>3d}) ║\n"
            
            report += "╚══════════════════════════════════════════════════════════════╝"
            
            return report
            
        except Exception as e:
            return f"Error generating z-score report: {str(e)}"
    
    def export_pattern_log(self, filename: str = None) -> str:
        """Export pattern log to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"brainwave_pattern_log_{timestamp}.json"
        
        export_data = {
            'baseline_stats': self.baseline_stats,
            'pattern_history': self.pattern_history,
            'pattern_log': self.pattern_log,
            'export_timestamp': datetime.now().isoformat(),
            'total_patterns': len(self.pattern_log),
            'z_score_threshold': self.z_score_threshold
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Pattern log exported to {filename}")
        return filename
    
    def visualize_z_score_trends(self, save_path: str = None):
        """Visualize z-score trends and patterns."""
        if not self.pattern_log:
            print("No pattern log data available for visualization")
            return
        
        # Extract data for visualization
        timestamps = []
        freq_z_scores = []
        amp_z_scores = []
        anomaly_flags = []
        
        for entry in self.pattern_log[-100:]:  # Last 100 entries
            timestamps.append(datetime.fromisoformat(entry['timestamp']))
            freq_z_scores.append(entry['z_scores'].get('frequency_z_score', 0))
            amp_z_scores.append(entry['z_scores'].get('amplitude_z_score', 0))
            anomaly_flags.append(entry['z_scores'].get('overall_anomaly', False))
        
        if not timestamps:
            print("No valid data for visualization")
            return
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Z-Score Brainwave Pattern Analysis', fontsize=16)
        
        # Frequency z-scores over time
        ax1.plot(timestamps, freq_z_scores, 'b-', marker='o', markersize=3)
        ax1.axhline(y=self.z_score_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        ax1.axhline(y=-self.z_score_threshold, color='r', linestyle='--', alpha=0.7)
        ax1.set_title('Frequency Z-Scores Over Time')
        ax1.set_ylabel('Z-Score')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Amplitude z-scores over time
        ax2.plot(timestamps, amp_z_scores, 'g-', marker='s', markersize=3)
        ax2.axhline(y=self.z_score_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        ax2.axhline(y=-self.z_score_threshold, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('Amplitude Z-Scores Over Time')
        ax2.set_ylabel('Z-Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Anomaly detection over time
        anomaly_y = [1 if flag else 0 for flag in anomaly_flags]
        ax3.fill_between(timestamps, anomaly_y, alpha=0.6, color='red', label='Anomalies')
        ax3.set_title('Anomaly Detection Timeline')
        ax3.set_ylabel('Anomaly Detected')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Z-score correlation
        ax4.scatter(freq_z_scores, amp_z_scores, 
                   c=['red' if flag else 'blue' for flag in anomaly_flags],
                   alpha=0.6, s=30)
        ax4.axvline(x=self.z_score_threshold, color='r', linestyle='--', alpha=0.7)
        ax4.axvline(x=-self.z_score_threshold, color='r', linestyle='--', alpha=0.7)
        ax4.axhline(y=self.z_score_threshold, color='r', linestyle='--', alpha=0.7)
        ax4.axhline(y=-self.z_score_threshold, color='r', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Frequency Z-Score')
        ax4.set_ylabel('Amplitude Z-Score')
        ax4.set_title('Z-Score Correlation (Red=Anomaly)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # ... (rest of the existing methods remain unchanged)
    
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
    
    def classify_frequency(self, frequency: float) -> str:
        """Classify frequency into brainwave categories."""
        for wave_type, (min_freq, max_freq) in self.FREQUENCY_RANGES.items():
            if min_freq <= frequency < max_freq:
                return wave_type
        return 'Unknown'
    
    def validate_data(self, data: Dict) -> bool:
        """Validate input data structure and values."""
        required_keys = ['pulseFrequency']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(data[key], (int, float)) or data[key] <= 0:
                raise ValueError(f"Invalid value for {key}: must be positive number")
        
        return True


# Enhanced usage example with z-score analysis
def main():
    """Main function demonstrating z-score brainwave pattern tracking."""
    analyzer = BrainwaveAnalyzer()
    
    print("=" * 80)
    print("Z-SCORE BRAINWAVE PATTERN ANALYZER")
    print("=" * 80)
    
    # Step 1: Generate baseline data
    print("Generating baseline data...")
    baseline_frequencies = np.random.normal(10.0, 2.0, 100)  # Alpha-dominant baseline
    baseline_amplitudes = np.random.normal(0.5, 0.1, 100)
    
    # Establish baseline
    analyzer.establish_baseline(baseline_frequencies.tolist(), baseline_amplitudes.tolist())
    print("Baseline established successfully!")
    
    # Step 2: Simulate real-time measurements with various patterns
    print("\nSimulating real-time brainwave measurements...")
    
    test_patterns = [
        # Normal readings
        {'pulseFrequency': 9.5, 'pulseAmplitude': 0.48},
        {'pulseFrequency': 10.2, 'pulseAmplitude': 0.52},
        {'pulseFrequency': 11.0, 'pulseAmplitude': 0.45},
        
        # Anomalous readings
        {'pulseFrequency': 15.5, 'pulseAmplitude': 0.75},  # High frequency/amplitude
        {'pulseFrequency': 5.2, 'pulseAmplitude': 0.25},   # Low frequency/amplitude
        {'pulseFrequency': 8.8, 'pulseAmplitude': 0.95},   # High amplitude only
        
        # Extreme readings
        {'pulseFrequency': 25.0, 'pulseAmplitude': 0.15},  # Very high frequency
        {'pulseFrequency': 1.5, 'pulseAmplitude': 0.85},   # Very low frequency
    ]
    
    for i, data in enumerate(test_patterns):
        print(f"\n--- Measurement {i+1} ---")
        report = analyzer.generate_z_score_report(data)
        print(report)
    
    # Step 3: Analyze pattern clusters
    print("\n" + "="*60)
    print("PATTERN CLUSTER ANALYSIS")
    print("="*60)
    clusters = analyzer.detect_pattern_clusters(60)
    print(f"Summary: {clusters['summary']}")
    
    for cluster in clusters['clusters']:
        print(f"\nCluster: {cluster['pattern_type']}")
        print(f"  Count: {cluster['count']}")
        print(f"  Duration: {cluster['duration_minutes']:.1f} minutes")
        print(f"  Significance Score: {cluster['significance_score']}")
    
    # Step 4: Export data and visualize
    print(f"\nExporting pattern log...")
    log_file = analyzer.export_pattern_log()
    print(f"Pattern log exported to: {log_file}")
    
    print(f"\nGenerating z-score trend visualization...")
    analyzer.visualize_z_score_trends()
    
    print(f"\nZ-score analysis complete!")

if __name__ == "__main__":
    main()
