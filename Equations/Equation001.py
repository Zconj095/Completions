import numpy as np
from typing import Callable, Dict, Any
import logging

class SDMRCalculator:
  """
  Self-Defined Memory Retrieval Calculator with enhanced functionality.
  """
  
  def __init__(self):
    self.logger = logging.getLogger(__name__)
    
  def calculate_sdmr(self, cdt: float, umn: float, cr: float, sci: float, 
            f_cdt_func: Callable[[float], float], 
            dot_product_func: Callable[[float, float, float], float],
            normalize: bool = False) -> Dict[str, Any]:
    """
    Calculates the Self-Defined Memory Retrieval (SDMR) score with detailed metrics.

    Args:
      cdt: Created Dictionary Terminology influence value
      umn: Utilization of Memory Management Notes (0-1)
      cr: Comprehension of Bodily Effects (0-1)
      sci: Self-Defining Critical Information (0-1)
      f_cdt_func: Function for CDT influence transformation
      dot_product_func: Function for weighted combination of UMN, CR, SCI
      normalize: Whether to normalize the final score (0-1 range)

    Returns:
      Dictionary containing SDMR score and component analysis
    """
    # Validate inputs
    self._validate_inputs(umn, cr, sci)
    
    # Apply CDT transformation
    f_cdt = f_cdt_func(cdt)
    
    # Calculate weighted combination
    dot_product = dot_product_func(umn, cr, sci)
    
    # Calculate base SDMR score
    sdmr = f_cdt * dot_product
    
    # Optional normalization
    normalized_sdmr = self._normalize_score(sdmr) if normalize else None
    
    return {
      'sdmr_score': sdmr,
      'normalized_score': normalized_sdmr,
      'f_cdt': f_cdt,
      'dot_product': dot_product,
      'components': {'cdt': cdt, 'umn': umn, 'cr': cr, 'sci': sci}
    }
  
  def _validate_inputs(self, umn: float, cr: float, sci: float) -> None:
    """Validate input parameters are within expected ranges."""
    for param, name in [(umn, 'UMN'), (cr, 'CR'), (sci, 'SCI')]:
      if not 0 <= param <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {param}")
  
  def _normalize_score(self, score: float) -> float:
    """Normalize score using sigmoid function."""
    return 1 / (1 + np.exp(-score))

# Predefined transformation functions
class TransformationFunctions:
  """Collection of common transformation functions."""
  
  @staticmethod
  def exponential(x: float, scale: float = 1.0) -> float:
    """Exponential transformation: scale * exp(x)"""
    return scale * np.exp(x)
  
  @staticmethod
  def sigmoid(x: float, steepness: float = 1.0) -> float:
    """Sigmoid transformation: 1 / (1 + exp(-steepness * x))"""
    return 1 / (1 + np.exp(-steepness * x))
  
  @staticmethod
  def linear(x: float, slope: float = 1.0, intercept: float = 0.0) -> float:
    """Linear transformation: slope * x + intercept"""
    return slope * x + intercept
  
  @staticmethod
  def power(x: float, exponent: float = 2.0) -> float:
    """Power transformation: x^exponent"""
    return np.power(abs(x), exponent) * np.sign(x)

class CombinationFunctions:
  """Collection of common combination functions."""
  
  @staticmethod
  def weighted_sum(umn: float, cr: float, sci: float, 
          weights: tuple = (1.0, 1.0, 1.0)) -> float:
    """Weighted sum: w1*umn + w2*cr + w3*sci"""
    return weights[0] * umn + weights[1] * cr + weights[2] * sci
  
  @staticmethod
  def geometric_mean(umn: float, cr: float, sci: float) -> float:
    """Geometric mean: (umn * cr * sci)^(1/3)"""
    return np.power(umn * cr * sci, 1/3)
  
  @staticmethod
  def harmonic_mean(umn: float, cr: float, sci: float) -> float:
    """Harmonic mean: 3 / (1/umn + 1/cr + 1/sci)"""
    return 3 / (1/umn + 1/cr + 1/sci) if all(x > 0 for x in [umn, cr, sci]) else 0

# Example usage with enhanced functionality
def main():
  # Initialize calculator
  calculator = SDMRCalculator()
  
  # Test parameters
  test_cases = [
    {'cdt': 5, 'umn': 0.8, 'cr': 0.7, 'sci': 0.9},
    {'cdt': 2, 'umn': 0.6, 'cr': 0.8, 'sci': 0.5},
    {'cdt': -1, 'umn': 0.9, 'cr': 0.6, 'sci': 0.7}
  ]
  
  # Test different function combinations
  function_combos = [
    ("Exponential + Weighted Sum", 
     TransformationFunctions.exponential,
     lambda u, c, s: CombinationFunctions.weighted_sum(u, c, s, (2, 1, 1))),
    
    ("Sigmoid + Geometric Mean",
     lambda x: TransformationFunctions.sigmoid(x, 0.5),
     CombinationFunctions.geometric_mean),
    
    ("Linear + Harmonic Mean",
     lambda x: TransformationFunctions.linear(x, 1.5, 0.5),
     CombinationFunctions.harmonic_mean)
  ]
  
  print("Enhanced SDMR Analysis")
  print("=" * 50)
  
  for i, params in enumerate(test_cases, 1):
    print(f"\nTest Case {i}: {params}")
    print("-" * 30)
    
    for name, f_cdt, dot_func in function_combos:
      try:
        result = calculator.calculate_sdmr(
          normalize=True,
          f_cdt_func=f_cdt,
          dot_product_func=dot_func,
          **params
        )
        
        print(f"{name}:")
        print(f"  SDMR Score: {result['sdmr_score']:.4f}")
        print(f"  Normalized: {result['normalized_score']:.4f}")
        print(f"  F(CDT): {result['f_cdt']:.4f}")
        print(f"  Dot Product: {result['dot_product']:.4f}")
        
      except Exception as e:
        print(f"{name}: Error - {e}")

if __name__ == "__main__":
  main()