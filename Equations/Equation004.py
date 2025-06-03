import math
import random
from typing import Dict, List, Tuple

class MemoryResponse:
    """Enhanced automatic memory response system with multiple retrieval algorithms."""
    
    def __init__(self, algorithm: str = "weighted_exponential"):
        self.algorithm = algorithm
        self.response_history: List[float] = []
        
    def automatic_memory_response(self, memory_trace: float, instincts: float, 
                                emotions: float, body_energy: float, 
                                consciousness: float, context: Dict = None) -> Dict:
        """
        Calculates enhanced automatic memory response with multiple algorithms.
        
        Args:
            memory_trace: Memory strength (0.0-1.0)
            instincts: Biological drives influence (0.0-1.0)
            emotions: Emotional state influence (-1.0 to 1.0, negative for suppression)
            body_energy: Physical/energetic well-being (0.0-1.0)
            consciousness: Conscious/subconscious influence (0.0-1.0)
            context: Optional contextual factors
            
        Returns:
            Dict containing AMR value, confidence, and breakdown
        """
        # Validate inputs
        self._validate_inputs(memory_trace, instincts, emotions, body_energy, consciousness)
        
        # Apply contextual modifiers
        if context:
            memory_trace, instincts, emotions, body_energy, consciousness = \
                self._apply_context(memory_trace, instincts, emotions, body_energy, consciousness, context)
        
        # Calculate AMR using selected algorithm
        amr = self._calculate_amr(memory_trace, instincts, emotions, body_energy, consciousness)
        
        # Calculate confidence and other metrics
        confidence = self._calculate_confidence(memory_trace, body_energy, consciousness)
        retrieval_speed = self._calculate_retrieval_speed(instincts, emotions, body_energy)
        
        # Store in history
        self.response_history.append(amr)
        
        return {
            'amr': round(amr, 4),
            'confidence': round(confidence, 4),
            'retrieval_speed': round(retrieval_speed, 4),
            'algorithm_used': self.algorithm,
            'factor_breakdown': {
                'memory_trace': memory_trace,
                'instincts': instincts,
                'emotions': emotions,
                'body_energy': body_energy,
                'consciousness': consciousness
            }
        }
    
    def _validate_inputs(self, m: float, i: float, e: float, b: float, c: float):
        """Validate input parameters are within expected ranges."""
        if not (0 <= m <= 1):
            raise ValueError("memory_trace must be between 0 and 1")
        if not (0 <= i <= 1):
            raise ValueError("instincts must be between 0 and 1")
        if not (-1 <= e <= 1):
            raise ValueError("emotions must be between -1 and 1")
        if not (0 <= b <= 1):
            raise ValueError("body_energy must be between 0 and 1")
        if not (0 <= c <= 1):
            raise ValueError("consciousness must be between 0 and 1")
    
    def _apply_context(self, m: float, i: float, e: float, b: float, c: float, context: Dict) -> Tuple:
        """Apply contextual modifiers to base factors."""
        stress_level = context.get('stress_level', 0)
        time_of_day = context.get('time_of_day', 'day')
        social_setting = context.get('social_setting', False)
        
        # Stress reduces memory trace and body energy
        m *= (1 - stress_level * 0.3)
        b *= (1 - stress_level * 0.4)
        
        # Time of day affects consciousness and body energy
        if time_of_day == 'night':
            c *= 0.8
            b *= 0.9
        elif time_of_day == 'morning':
            c *= 1.1
            b *= 1.05
            
        # Social setting can enhance instincts
        if social_setting:
            i *= 1.2
            
        return m, i, e, b, c
    
    def _calculate_amr(self, m: float, i: float, e: float, b: float, c: float) -> float:
        """Calculate AMR using the selected algorithm."""
        algorithms = {
            'simple_sum': lambda: m + i + abs(e) + b + c,
            'weighted_average': lambda: (m * 0.3 + i * 0.15 + abs(e) * 0.25 + b * 0.2 + c * 0.1),
            'exponential': lambda: math.exp((m + i + abs(e) + b + c) / 5) - 1,
            'weighted_exponential': lambda: math.exp((m * 0.4 + i * 0.1 + abs(e) * 0.3 + b * 0.15 + c * 0.05)) - 1,
            'harmonic': lambda: 5 / (1/max(m, 0.01) + 1/max(i, 0.01) + 1/max(abs(e), 0.01) + 1/max(b, 0.01) + 1/max(c, 0.01)),
            'geometric': lambda: (m * i * abs(e) * b * c) ** 0.2,
            'neural_network': lambda: self._neural_approximation(m, i, e, b, c)
        }
        
        return algorithms.get(self.algorithm, algorithms['weighted_exponential'])()
    
    def _neural_approximation(self, m: float, i: float, e: float, b: float, c: float) -> float:
        """Simple neural network approximation for memory retrieval."""
        # Hidden layer weights (simplified)
        w1 = [0.3, 0.2, 0.25, 0.15, 0.1]
        w2 = [0.4, -0.1, 0.3, 0.2, 0.1]
        w3 = [0.2, 0.3, 0.1, 0.25, 0.15]
        
        inputs = [m, i, e, b, c]
        
        # Hidden layer activations
        h1 = math.tanh(sum(w * x for w, x in zip(w1, inputs)))
        h2 = math.tanh(sum(w * x for w, x in zip(w2, inputs)))
        h3 = math.tanh(sum(w * x for w, x in zip(w3, inputs)))
        
        # Output layer
        output_weights = [0.4, 0.35, 0.25]
        return max(0, sum(w * h for w, h in zip(output_weights, [h1, h2, h3])))
    
    def _calculate_confidence(self, m: float, b: float, c: float) -> float:
        """Calculate confidence in the memory response."""
        return (m * 0.5 + b * 0.3 + c * 0.2)
    
    def _calculate_retrieval_speed(self, i: float, e: float, b: float) -> float:
        """Calculate how quickly the memory can be retrieved."""
        return (i * 0.4 + abs(e) * 0.3 + b * 0.3)
    
    def get_statistics(self) -> Dict:
        """Get statistics about memory response history."""
        if not self.response_history:
            return {}
        
        return {
            'mean_amr': sum(self.response_history) / len(self.response_history),
            'max_amr': max(self.response_history),
            'min_amr': min(self.response_history),
            'total_responses': len(self.response_history)
        }

# Example usage with enhanced features
if __name__ == "__main__":
    # Create memory response system
    memory_system = MemoryResponse(algorithm="weighted_exponential")
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'Normal recall',
            'params': (0.8, 0.2, 0.5, 0.7, 0.3),
            'context': {}
        },
        {
            'name': 'Stressed recall',
            'params': (0.8, 0.4, -0.3, 0.5, 0.6),
            'context': {'stress_level': 0.7, 'time_of_day': 'night'}
        },
        {
            'name': 'Social memory',
            'params': (0.6, 0.6, 0.8, 0.8, 0.4),
            'context': {'social_setting': True, 'time_of_day': 'morning'}
        }
    ]
    
    for scenario in scenarios:
        result = memory_system.automatic_memory_response(*scenario['params'], context=scenario['context'])
        print(f"\n{scenario['name']}:")
        print(f"AMR: {result['amr']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Retrieval Speed: {result['retrieval_speed']}")
    
    # Display statistics
    print(f"\nSystem Statistics: {memory_system.get_statistics()}")
