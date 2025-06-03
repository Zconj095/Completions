import numpy as np
import copy
from typing import Tuple, List, Optional
from generate_network_topology import generate_network_topology

class NodeFailureSimulator:
    """Enhanced node failure simulator with recovery and analysis capabilities."""
    
    def __init__(self, original_topology: np.ndarray):
        """Initialize with the original network topology."""
        self.original_topology = original_topology.copy()
        self.current_topology = original_topology.copy()
        self.failed_nodes = set()
        self.failure_history = []
    
    def simulate_random_failures(self, num_failures: int = 1) -> Tuple[np.ndarray, List[int]]:
        """Simulate random node failures."""
        num_nodes = self.current_topology.shape[0]
        available_nodes = [i for i in range(num_nodes) if i not in self.failed_nodes]
        
        if num_failures > len(available_nodes):
            raise ValueError(f"Cannot fail {num_failures} nodes. Only {len(available_nodes)} nodes available.")
        
        new_failed_nodes = np.random.choice(available_nodes, size=num_failures, replace=False).tolist()
        return self._apply_failures(new_failed_nodes)
    
    def simulate_targeted_failures(self, target_nodes: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Simulate failures of specific nodes."""
        available_nodes = [node for node in target_nodes if node not in self.failed_nodes]
        return self._apply_failures(available_nodes)
    
    def simulate_cascading_failures(self, initial_failures: int = 1, threshold: float = 0.5) -> Tuple[np.ndarray, List[int]]:
        """Simulate cascading failures based on connectivity threshold."""
        # Initial random failures
        self.simulate_random_failures(initial_failures)
        
        cascaded_nodes = []
        while True:
            # Find nodes that fall below connectivity threshold
            new_failures = []
            for node in range(self.current_topology.shape[0]):
                if node not in self.failed_nodes:
                    original_connections = np.sum(self.original_topology[node, :] > 0)
                    current_connections = np.sum(self.current_topology[node, :] > 0)
                    
                    if original_connections > 0 and (current_connections / original_connections) < threshold:
                        new_failures.append(node)
            
            if not new_failures:
                break
                
            self._apply_failures(new_failures)
            cascaded_nodes.extend(new_failures)
        
        return self.current_topology, cascaded_nodes
    
    def _apply_failures(self, nodes_to_fail: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Apply failures to specified nodes."""
        for node in nodes_to_fail:
            if node not in self.failed_nodes:
                # Record failure
                self.failed_nodes.add(node)
                self.failure_history.append(node)
                
                # Disable connections
                self.current_topology[node, :] = 0
                self.current_topology[:, node] = 0
        
        return self.current_topology, nodes_to_fail
    
    def recover_node(self, node: int) -> bool:
        """Recover a failed node."""
        if node in self.failed_nodes:
            self.failed_nodes.remove(node)
            # Restore original connections
            self.current_topology[node, :] = self.original_topology[node, :]
            self.current_topology[:, node] = self.original_topology[:, node]
            return True
        return False
    
    def get_network_metrics(self) -> dict:
        """Calculate network resilience metrics."""
        num_nodes = self.original_topology.shape[0]
        active_nodes = num_nodes - len(self.failed_nodes)
        
        # Calculate connectivity
        connected_components = self._count_connected_components()
        
        # Calculate average path length for largest component
        largest_component_size = max([len(comp) for comp in connected_components]) if connected_components else 0
        
        return {
            'total_nodes': num_nodes,
            'active_nodes': active_nodes,
            'failed_nodes': len(self.failed_nodes),
            'failure_rate': len(self.failed_nodes) / num_nodes,
            'connected_components': len(connected_components),
            'largest_component_size': largest_component_size,
            'network_fragmentation': len(connected_components) > 1
        }
    
    def _count_connected_components(self) -> List[List[int]]:
        """Count connected components in current topology."""
        num_nodes = self.current_topology.shape[0]
        visited = [False] * num_nodes
        components = []
        
        for node in range(num_nodes):
            if not visited[node] and node not in self.failed_nodes:
                component = []
                self._dfs(node, visited, component)
                if component:
                    components.append(component)
        
        return components
    
    def _dfs(self, node: int, visited: List[bool], component: List[int]):
        """Depth-first search for connected components."""
        visited[node] = True
        component.append(node)
        
        for neighbor in range(self.current_topology.shape[0]):
            if (not visited[neighbor] and 
                self.current_topology[node, neighbor] > 0 and 
                neighbor not in self.failed_nodes):
                self._dfs(neighbor, visited, component)
    
    def reset(self):
        """Reset to original topology."""
        self.current_topology = self.original_topology.copy()
        self.failed_nodes.clear()
        self.failure_history.clear()

# Enhanced example usage
if __name__ == "__main__":
    # Generate the network topology
    network_topology = generate_network_topology(10)
    
    # Initialize simulator
    simulator = NodeFailureSimulator(network_topology)
    
    print("Original Network Topology:")
    print(network_topology)
    print(f"Network Metrics: {simulator.get_network_metrics()}")
    
    # Random failures
    print("\n--- Random Failures ---")
    modified_topology, failed = simulator.simulate_random_failures(2)
    print(f"Failed nodes: {failed}")
    print(f"Network Metrics: {simulator.get_network_metrics()}")
    
    # Reset and try cascading failures
    simulator.reset()
    print("\n--- Cascading Failures ---")
    _, cascaded = simulator.simulate_cascading_failures(1, threshold=0.3)
    print(f"Cascaded failures: {cascaded}")
    print(f"Network Metrics: {simulator.get_network_metrics()}")
    
    # Node recovery
    if simulator.failed_nodes:
        recovered_node = list(simulator.failed_nodes)[0]
        simulator.recover_node(recovered_node)
        print(f"\nRecovered node {recovered_node}")
        print(f"Network Metrics: {simulator.get_network_metrics()}")
