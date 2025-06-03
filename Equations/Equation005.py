import math
from typing import Callable, Union, Optional
from dataclasses import dataclass

@dataclass
class DivinityMetrics:
    """Data class to hold divinity-related measurements."""
    divine_mark: float
    divine_power: float
    other_memory: float
    
    def __post_init__(self):
        """Validate inputs are within valid ranges."""
        for field_name, value in [("divine_mark", self.divine_mark), 
                                 ("divine_power", self.divine_power), 
                                 ("other_memory", self.other_memory)]:
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")

class HolyMemoryCalculator:
    """Calculator for divine memory influence with multiple probability functions."""
    
    @staticmethod
    def sigmoid_function(x: float, steepness: float = 10) -> float:
        """Sigmoid function for smooth probability transition."""
        return 1 / (1 + math.exp(-steepness * (x - 0.5)))
    
    @staticmethod
    def exponential_function(x: float, power: float = 2) -> float:
        """Exponential function for accelerating probability."""
        return min(x ** power, 1.0)
    
    @staticmethod
    def threshold_function(x: float, threshold: float = 0.5) -> float:
        """Step function with customizable threshold."""
        return 1.0 if x >= threshold else 0.0
    
    @staticmethod
    def linear_function(x: float) -> float:
        """Simple linear probability function."""
        return min(x, 1.0)

def calculate_holy_memory(
    metrics: DivinityMetrics, 
    probability_func: Callable[[float], float],
    influence_factor: float = 1.0
) -> dict:
    """
    Enhanced calculation of divine memory influence with detailed output.
    
    Args:
        metrics: DivinityMetrics object containing divine attributes
        probability_func: Function to calculate holiness probability
        influence_factor: Multiplier for divine influence strength
    
    Returns:
        Dictionary containing detailed calculation results
    """
    
    # Calculate divine influence
    divine_influence = metrics.divine_mark * metrics.divine_power * influence_factor
    divine_influence = min(divine_influence, 1.0)  # Cap at 1.0
    
    # Calculate probability of holiness
    probability_holy = probability_func(divine_influence)
    
    # Calculate final holy memory value
    holy_memory_value = probability_holy * 1.0 + (1 - probability_holy) * metrics.other_memory
    
    return {
        "holy_memory": holy_memory_value,
        "divine_influence": divine_influence,
        "probability_holy": probability_holy,
        "memory_composition": {
            "divine_portion": probability_holy,
            "other_portion": 1 - probability_holy,
            "other_memory_contribution": (1 - probability_holy) * metrics.other_memory
        }
    }

def demonstrate_calculations():
    """Demonstrate the enhanced holy memory calculator with different functions."""
    
    # Create sample metrics
    metrics = DivinityMetrics(
        divine_mark=0.8,
        divine_power=0.9,
        other_memory=0.2
    )
    
    calculator = HolyMemoryCalculator()
    
    # Test different probability functions
    functions = {
        "Exponential": calculator.exponential_function,
        "Sigmoid": calculator.sigmoid_function,
        "Threshold": calculator.threshold_function,
        "Linear": calculator.linear_function
    }
    
    print("=== Holy Memory Calculation Results ===\n")
    print(f"Input Metrics: Divine Mark={metrics.divine_mark}, "
          f"Divine Power={metrics.divine_power}, Other Memory={metrics.other_memory}\n")
    
    for func_name, func in functions.items():
        result = calculate_holy_memory(metrics, func)
        
        print(f"--- {func_name} Function ---")
        print(f"Holy Memory Value: {result['holy_memory']:.4f}")
        print(f"Divine Influence: {result['divine_influence']:.4f}")
        print(f"Probability Holy: {result['probability_holy']:.4f}")
        print(f"Divine Portion: {result['memory_composition']['divine_portion']:.4f}")
        print(f"Other Memory Contribution: {result['memory_composition']['other_memory_contribution']:.4f}")
        print()

if __name__ == "__main__":
    demonstrate_calculations()
