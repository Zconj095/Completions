import numpy as np
from typing import Union, Tuple
import warnings

class MemoryProcessor:
    """
    A class to model memory transformation based on psychological factors.
    """
    
    def __init__(self, random_seed: int = None):
        """Initialize the memory processor with optional random seed for reproducibility."""
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def impure_memory(self, M: np.ndarray, D: np.ndarray, G: float, 
                     AS: float, MS: float, CR: float) -> Tuple[float, dict]:
        """
        Model memory transformation based on desires and psychological biases.
        
        Args:
            M (np.ndarray): Memory components
            D (np.ndarray): Desires array
            G (float): Goodwill/Faith factor
            AS (float): Automatic Subjection factor
            MS (float): Manual Subjection factor
            CR (float): Chemical Response factor
            
        Returns:
            Tuple[float, dict]: Impurity score and detailed breakdown
            
        Raises:
            ValueError: If input arrays have incompatible dimensions or invalid values
        """
        # Input validation
        self._validate_inputs(M, D, G, AS, MS, CR)
        
        # Normalize desire weights to prevent division by zero
        desire_sum = np.sum(D)
        if desire_sum == 0:
            warnings.warn("All desires are zero, using uniform weights")
            desire_weights = np.ones_like(D) / len(D)
        else:
            desire_weights = D / desire_sum
        
        # Calculate memory distortions
        distortion_base = np.dot(desire_weights, np.random.randn(len(M)))
        random_bias = AS * np.random.rand()
        
        # Apply transformations to memory
        biased_memory = M + distortion_base + random_bias + MS
        
        # Calculate psychological factors
        destructive_score = max(0, np.max(D) - G)  # Prevent negative scores
        memory_variance = np.var(biased_memory)
        
        # Compute impurity components
        memory_component = np.mean(biased_memory)
        destructive_component = destructive_score * CR
        variance_component = memory_variance * 0.1  # Small weight for variance
        
        # Overall impurity score
        impurity = memory_component + destructive_component + variance_component
        
        # Detailed breakdown for analysis
        breakdown = {
            'memory_component': memory_component,
            'destructive_component': destructive_component,
            'variance_component': variance_component,
            'biased_memory': biased_memory,
            'desire_weights': desire_weights,
            'destructive_score': destructive_score
        }
        
        return impurity, breakdown
    
    def _validate_inputs(self, M: np.ndarray, D: np.ndarray, G: float, 
                        AS: float, MS: float, CR: float) -> None:
        """Validate input parameters."""
        if not isinstance(M, np.ndarray) or not isinstance(D, np.ndarray):
            raise ValueError("M and D must be numpy arrays")
        
        if len(M) != len(D):
            raise ValueError("Memory and Desires arrays must have same length")
        
        if len(M) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in [G, AS, MS, CR]):
            raise ValueError("G, AS, MS, CR must be numeric values")
        
        if np.any(D < 0):
            raise ValueError("Desires must be non-negative")
    
    def analyze_memory_patterns(self, M: np.ndarray, D: np.ndarray, G: float, 
                               AS: float, MS: float, CR: float, 
                               num_iterations: int = 100) -> dict:
        """
        Analyze memory patterns over multiple iterations.
        
        Returns:
            dict: Statistical analysis of memory transformations
        """
        scores = []
        memory_means = []
        destructive_scores = []
        
        for _ in range(num_iterations):
            score, breakdown = self.impure_memory(M, D, G, AS, MS, CR)
            scores.append(score)
            memory_means.append(breakdown['memory_component'])
            destructive_scores.append(breakdown['destructive_score'])
        
        return {
            'mean_impurity': np.mean(scores),
            'std_impurity': np.std(scores),
            'min_impurity': np.min(scores),
            'max_impurity': np.max(scores),
            'mean_memory_component': np.mean(memory_means),
            'mean_destructive_component': np.mean(destructive_scores)
        }

# Example usage with enhanced functionality
def main():
    # Initialize processor
    processor = MemoryProcessor(random_seed=42)
    
    # Example data
    M = np.array([0.7, 0.8, 0.5, 0.6])  # Memory components
    D = np.array([0.3, 0.5, 0.2, 0.4])  # Desires
    G = 0.1  # Goodwill/Faith
    AS = 0.2  # Automatic Subjection
    MS = 0.1  # Manual Subjection
    CR = 1.2  # Chemical Response factor
    
    # Single calculation
    impurity_score, breakdown = processor.impure_memory(M, D, G, AS, MS, CR)
    
    print("=== Single Memory Analysis ===")
    print(f"Impure memory score: {impurity_score:.4f}")
    print(f"Memory component: {breakdown['memory_component']:.4f}")
    print(f"Destructive component: {breakdown['destructive_component']:.4f}")
    print(f"Variance component: {breakdown['variance_component']:.4f}")
    
    # Pattern analysis
    analysis = processor.analyze_memory_patterns(M, D, G, AS, MS, CR)
    
    print("\n=== Pattern Analysis (100 iterations) ===")
    for key, value in analysis.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
