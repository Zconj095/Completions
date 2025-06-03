import math
import numpy as np
from typing import Union, List, Tuple
from dataclasses import dataclass
from enum import Enum

class MemoryType(Enum):
  """Different types of memory processing modes."""
  LINEAR = "linear"
  EXPONENTIAL = "exponential"
  LOGARITHMIC = "logarithmic"
  SIGMOID = "sigmoid"

@dataclass
class MemoryMetrics:
  """Container for memory calculation results."""
  score: float
  efficiency_rating: str
  bottleneck_factor: str
  optimization_suggestions: List[str]

class MicroManagedMemory:
  """
  Advanced micromanaged memory calculator with multiple processing modes
  and comprehensive analysis capabilities.
  """
  
  def __init__(self, processing_mode: MemoryType = MemoryType.EXPONENTIAL):
    self.processing_mode = processing_mode
    self.history = []
  
  def calculate(self, 
         data_density: float, 
         temporal_resolution: float, 
         contextual_awareness: float, 
         network_efficiency: float,
         cognitive_load: float = 1.0) -> MemoryMetrics:
    """
    Enhanced memory calculation with multiple processing modes.
    
    Args:
      data_density: Information complexity (0.1-100)
      temporal_resolution: Access precision in seconds (0.001-10)
      contextual_awareness: Relationship understanding (0-1)
      network_efficiency: Traversal speed (0.1-10)
      cognitive_load: Mental processing burden (0.1-5)
    
    Returns:
      MemoryMetrics object with comprehensive analysis
    """
    
    # Validate inputs
    self._validate_inputs(data_density, temporal_resolution, 
              contextual_awareness, network_efficiency, cognitive_load)
    
    # Calculate base function based on processing mode
    base_function = self._calculate_base_function(
      data_density, temporal_resolution, contextual_awareness
    )
    
    # Apply network efficiency and cognitive load
    memory_score = (base_function * network_efficiency) / cognitive_load
    
    # Store calculation in history
    self.history.append({
      'inputs': (data_density, temporal_resolution, contextual_awareness, 
            network_efficiency, cognitive_load),
      'score': memory_score,
      'mode': self.processing_mode.value
    })
    
    # Generate comprehensive metrics
    return self._generate_metrics(memory_score, data_density, temporal_resolution,
                  contextual_awareness, network_efficiency, cognitive_load)
  
  def _calculate_base_function(self, dd: float, tr: float, ca: float) -> float:
    """Calculate base function using selected processing mode."""
    
    if self.processing_mode == MemoryType.LINEAR:
      return dd * tr * ca
    
    elif self.processing_mode == MemoryType.EXPONENTIAL:
      return math.pow(dd * tr * ca, 0.7)
    
    elif self.processing_mode == MemoryType.LOGARITHMIC:
      return math.log(1 + dd * tr * ca)
    
    elif self.processing_mode == MemoryType.SIGMOID:
      x = dd * tr * ca
      return 2 / (1 + math.exp(-x/10)) - 1
    
    return dd * tr * ca  # fallback
  
  def _validate_inputs(self, dd: float, tr: float, ca: float, ne: float, cl: float):
    """Validate input parameters."""
    if not (0.1 <= dd <= 100):
      raise ValueError("Data density must be between 0.1 and 100")
    if not (0.001 <= tr <= 10):
      raise ValueError("Temporal resolution must be between 0.001 and 10")
    if not (0 <= ca <= 1):
      raise ValueError("Contextual awareness must be between 0 and 1")
    if not (0.1 <= ne <= 10):
      raise ValueError("Network efficiency must be between 0.1 and 10")
    if not (0.1 <= cl <= 5):
      raise ValueError("Cognitive load must be between 0.1 and 5")
  
  def _generate_metrics(self, score: float, dd: float, tr: float, 
             ca: float, ne: float, cl: float) -> MemoryMetrics:
    """Generate comprehensive analysis metrics."""
    
    # Efficiency rating
    if score > 10:
      efficiency = "Excellent"
    elif score > 5:
      efficiency = "Good"
    elif score > 2:
      efficiency = "Average"
    else:
      efficiency = "Poor"
    
    # Identify bottleneck
    factors = {
      'data_density': dd,
      'temporal_resolution': tr,
      'contextual_awareness': ca,
      'network_efficiency': ne,
      'cognitive_load': 1/cl  # Inverse since higher load is worse
    }
    bottleneck = min(factors, key=factors.get)
    
    # Generate optimization suggestions
    suggestions = []
    if ca < 0.5:
      suggestions.append("Improve contextual awareness through pattern recognition")
    if ne < 2:
      suggestions.append("Optimize network efficiency with better algorithms")
    if cl > 2:
      suggestions.append("Reduce cognitive load through task automation")
    if tr > 1:
      suggestions.append("Enhance temporal resolution with faster access methods")
    
    return MemoryMetrics(
      score=round(score, 3),
      efficiency_rating=efficiency,
      bottleneck_factor=bottleneck.replace('_', ' ').title(),
      optimization_suggestions=suggestions
    )
  
  def batch_calculate(self, parameter_sets: List[Tuple]) -> List[MemoryMetrics]:
    """Calculate memory metrics for multiple parameter sets."""
    return [self.calculate(*params) for params in parameter_sets]
  
  def get_optimization_report(self) -> str:
    """Generate a comprehensive optimization report."""
    if not self.history:
      return "No calculations performed yet."
    
    avg_score = np.mean([calc['score'] for calc in self.history])
    best_score = max([calc['score'] for calc in self.history])
    
    report = f"""
Micromanaged Memory Analysis Report
==================================
Calculations performed: {len(self.history)}
Average score: {avg_score:.3f}
Best score: {best_score:.3f}
Processing mode: {self.processing_mode.value}

Performance trend: {'Improving' if len(self.history) > 1 and self.history[-1]['score'] > self.history[0]['score'] else 'Stable'}
    """
    
    return report.strip()

# Enhanced example usage
def main():
  # Create memory calculator with different processing modes
  memory_calc = MicroManagedMemory(MemoryType.EXPONENTIAL)
  
  # Single calculation
  result = memory_calc.calculate(
    data_density=15,
    temporal_resolution=0.05,
    contextual_awareness=0.9,
    network_efficiency=3.5,
    cognitive_load=1.2
  )
  
  print(f"Memory Score: {result.score}")
  print(f"Efficiency: {result.efficiency_rating}")
  print(f"Bottleneck: {result.bottleneck_factor}")
  print(f"Suggestions: {', '.join(result.optimization_suggestions)}")
  
  # Batch processing example
  parameter_sets = [
    (10, 0.1, 0.8, 2.0, 1.0),
    (20, 0.05, 0.9, 3.0, 1.5),
    (5, 0.2, 0.6, 1.5, 0.8)
  ]
  
  batch_results = memory_calc.batch_calculate(parameter_sets)
  print(f"\nBatch processing complete. Results: {[r.score for r in batch_results]}")
  
  # Generate optimization report
  print("\n" + memory_calc.get_optimization_report())

if __name__ == "__main__":
  main()
