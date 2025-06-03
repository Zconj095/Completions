import numpy as np
from typing import Callable, Optional, Union
import matplotlib.pyplot as plt

class MemorySubjection:
    """
    A class for calculating memory subjection with various retrieval functions
    and enhanced functionality.
    """
    
    def __init__(self, decay_factor: float = 0.95, noise_level: float = 0.01):
        """
        Initialize the MemorySubjection calculator.
        
        Args:
            decay_factor: Memory decay factor (0-1)
            noise_level: Random noise level for realistic memory retrieval
        """
        self.decay_factor = decay_factor
        self.noise_level = noise_level
        self.history = []
    
    def memory_subjection(self, 
                         m: np.ndarray, 
                         i: np.ndarray, 
                         s: np.ndarray, 
                         f: Callable[[np.ndarray], np.ndarray],
                         time_step: Optional[int] = None) -> np.ndarray:
        """
        Calculates enhanced memory subjection with decay and noise.

        Args:
            m: Original memory (numpy array)
            i: Internal subjections (numpy array)
            s: External subjections (numpy array)
            f: Retrieval function
            time_step: Optional time step for temporal decay

        Returns:
            ms: Memory subjection (numpy array)
        """
        # Validate inputs
        if not all(isinstance(arr, np.ndarray) for arr in [m, i, s]):
            raise TypeError("All inputs must be numpy arrays")
        
        if not all(arr.shape == m.shape for arr in [i, s]):
            raise ValueError("All arrays must have the same shape")
        
        # Apply temporal decay if time_step is provided
        if time_step is not None:
            m = m * (self.decay_factor ** time_step)
        
        # Calculate weighted interaction between memory and external influences
        interaction = np.multiply(m, s)  # Element-wise for better control
        
        # Add memory strength weighting
        memory_strength = np.linalg.norm(m)
        weighted_interaction = interaction * memory_strength
        
        # Combine internal and external influences with nonlinear mixing
        combined_influences = i + weighted_interaction + self._interference_term(i, s)
        
        # Add realistic noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, combined_influences.shape)
            combined_influences += noise
        
        # Apply the retrieval function
        ms = f(combined_influences)
        
        # Store in history for analysis
        self.history.append({
            'memory': m.copy(),
            'internal': i.copy(),
            'external': s.copy(),
            'result': ms.copy(),
            'time_step': time_step
        })
        
        return ms
    
    def _interference_term(self, i: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Calculate interference between internal and external subjections."""
        return 0.1 * np.sin(i) * np.cos(s)
    
    def get_history(self) -> list:
        """Return the calculation history."""
        return self.history
    
    def clear_history(self):
        """Clear the calculation history."""
        self.history.clear()
    
    def plot_history(self, component: str = 'result'):
        """Plot the history of a specific component."""
        if not self.history:
            print("No history to plot")
            return
        
        data = [entry[component] for entry in self.history]
        plt.figure(figsize=(10, 6))
        for i in range(len(data[0])):
            values = [d[i] for d in data]
            plt.plot(values, label=f'Component {i}')
        
        plt.title(f'Memory Subjection History - {component.title()}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Enhanced retrieval functions
class RetrievalFunctions:
    """Collection of retrieval functions for memory subjection."""
    
    @staticmethod
    def sigmoid(x: np.ndarray, steepness: float = 1.0) -> np.ndarray:
        """Sigmoid activation with adjustable steepness."""
        return 1 / (1 + np.exp(-steepness * x))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation."""
        return np.tanh(x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit."""
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax for probability distribution."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian activation function."""
        return np.exp(-0.5 * (x / sigma) ** 2)

# Example usage with enhanced features
if __name__ == "__main__":
    # Initialize the memory subjection calculator
    ms_calc = MemorySubjection(decay_factor=0.95, noise_level=0.05)
    
    # Example data
    m = np.array([0.8, 0.6, 0.4])  # Strong initial memory
    i = np.array([0.1, 0.2, 0.3])  # Internal subjections
    s = np.array([0.4, 0.5, 0.6])  # External subjections
    
    print("Enhanced Memory Subjection Analysis")
    print("=" * 40)
    
    # Test different retrieval functions
    functions = {
        'Sigmoid': RetrievalFunctions.sigmoid,
        'Tanh': RetrievalFunctions.tanh,
        'ReLU': RetrievalFunctions.relu,
        'Softmax': RetrievalFunctions.softmax,
        'Gaussian': RetrievalFunctions.gaussian
    }
    
    for name, func in functions.items():
        ms = ms_calc.memory_subjection(m, i, s, func)
        print(f"{name:10}: {ms}")
    
    # Simulate temporal decay over multiple time steps
    print("\nTemporal Evolution:")
    print("-" * 20)
    
    ms_calc.clear_history()
    for t in range(5):
        ms = ms_calc.memory_subjection(m, i, s, RetrievalFunctions.sigmoid, time_step=t)
        print(f"Time {t:2d}: {ms}")
    
    # Batch processing for efficiency
    def batch_process(memories: list, internals: list, externals: list, 
                     func: Callable, time_steps: Optional[list] = None) -> list:
        """Process multiple memory subjections in batch."""
        results = []
        for idx, (mem, int_subj, ext_subj) in enumerate(zip(memories, internals, externals)):
            t_step = time_steps[idx] if time_steps else None
            result = ms_calc.memory_subjection(mem, int_subj, ext_subj, func, t_step)
            results.append(result)
        return results
    
    # Example batch processing
    batch_memories = [np.random.rand(3) for _ in range(3)]
    batch_internals = [np.random.rand(3) for _ in range(3)]
    batch_externals = [np.random.rand(3) for _ in range(3)]
    
    batch_results = batch_process(batch_memories, batch_internals, batch_externals, 
                                 RetrievalFunctions.sigmoid)
    
    print(f"\nBatch processed {len(batch_results)} memory subjections")
