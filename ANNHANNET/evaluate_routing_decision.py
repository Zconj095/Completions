import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
import time
from typing import Dict, List, Tuple

class NetworkRoutingEvaluator:
    def __init__(self, model_path: str = 'ann_hanet_model.h5'):
        """Initialize the routing evaluator with a trained model."""
        self.model = load_model(model_path)
        self.evaluation_history = []
        
    def generate_network_state(self, num_nodes: int = 5, connectivity_prob: float = 0.7) -> np.ndarray:
        """Generate a more realistic network state with controlled connectivity."""
        # Create adjacency matrix with probability-based connections
        network_state = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[1-connectivity_prob, connectivity_prob])
        
        # Ensure no self-loops
        np.fill_diagonal(network_state, 0)
        
        # Make symmetric (undirected graph)
        network_state = np.maximum(network_state, network_state.T)
        
        return network_state.reshape(1, -1)
    
    def calculate_metrics(self, predicted_route: np.ndarray, network_state: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive routing metrics."""
        route_binary = (predicted_route > 0.5).astype(int)
        
        metrics = {
            'efficiency_score': np.sum(predicted_route),
            'path_length': np.sum(route_binary),
            'confidence': np.mean(np.maximum(predicted_route, 1 - predicted_route)),
            'network_utilization': np.sum(route_binary) / np.sum(network_state),
            'route_diversity': len(np.unique(route_binary))
        }
        
        return metrics
    
    def evaluate_routing_decision(self, num_nodes: int = 5, num_evaluations: int = 1) -> List[Dict]:
        """Enhanced routing decision evaluation with multiple metrics."""
        results = []
        
        for i in range(num_evaluations):
            # Generate network state
            network_state = self.generate_network_state(num_nodes)
            
            # Predict route
            start_time = time.time()
            predicted_route = self.model.predict(network_state, verbose=0)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted_route, network_state)
            metrics['prediction_time'] = prediction_time
            
            # Store result
            result = {
                'evaluation_id': i + 1,
                'network_state': network_state.reshape(num_nodes, num_nodes),
                'predicted_route': predicted_route,
                'metrics': metrics
            }
            
            results.append(result)
            self.evaluation_history.append(result)
            
            # Print detailed results
            print(f"\n--- Evaluation {i + 1} ---")
            print(f"Network State:\n{result['network_state']}")
            print(f"Predicted Route: {predicted_route.flatten()}")
            for metric, value in metrics.items():
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        return results
    
    def batch_evaluate(self, num_nodes: int = 5, batch_size: int = 10) -> Dict[str, float]:
        """Perform batch evaluation and return summary statistics."""
        results = self.evaluate_routing_decision(num_nodes, batch_size)
        
        # Calculate summary statistics
        metrics_summary = {}
        for metric in results[0]['metrics'].keys():
            values = [r['metrics'][metric] for r in results]
            metrics_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return metrics_summary
    
    def visualize_performance(self, save_plot: bool = True):
        """Visualize routing performance over time."""
        if not self.evaluation_history:
            print("No evaluation history available.")
            return
        
        metrics = ['efficiency_score', 'path_length', 'confidence', 'prediction_time']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [eval_result['metrics'][metric] for eval_result in self.evaluation_history]
            axes[i].plot(values, marker='o')
            axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
            axes[i].set_xlabel('Evaluation')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].grid(True)
        
        plt.tight_layout()
        if save_plot:
            plt.savefig('routing_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = 'evaluation_results.json'):
        """Save evaluation results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = []
        for result in self.evaluation_history:
            # Convert metrics values to native Python types
            serializable_metrics = {k: float(v) for k, v in result['metrics'].items()}
            
            serializable_result = {
                'evaluation_id': result['evaluation_id'],
                'network_state': result['network_state'].tolist(),
                'predicted_route': result['predicted_route'].tolist(),
                'metrics': serializable_metrics
            }
            serializable_history.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"Results saved to {filename}")

# Usage example
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = NetworkRoutingEvaluator()
    
    # Single evaluation
    print("=== Single Evaluation ===")
    evaluator.evaluate_routing_decision(num_nodes=5)
    
    # Batch evaluation
    print("\n=== Batch Evaluation Summary ===")
    summary = evaluator.batch_evaluate(num_nodes=5, batch_size=5)
    for metric, stats in summary.items():
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Visualize and save results
    evaluator.visualize_performance()
    evaluator.save_results()
