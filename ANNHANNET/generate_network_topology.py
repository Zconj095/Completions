import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple

def generate_network_topology(
    num_nodes: int, 
    topology_type: str = "random", 
    connection_probability: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates various types of network topologies.

    Parameters:
    - num_nodes: int, the number of nodes in the network
    - topology_type: str, type of topology ('random', 'ring', 'star', 'complete', 'small_world')
    - connection_probability: float, probability of connection for random/small_world (0.0-1.0)
    - seed: Optional[int], random seed for reproducibility

    Returns:
    - A numpy array representing the adjacency matrix of the network
    """
    if seed is not None:
        np.random.seed(seed)
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    if topology_type == "random":
        return _generate_random_topology(num_nodes, connection_probability)
    elif topology_type == "ring":
        return _generate_ring_topology(num_nodes)
    elif topology_type == "star":
        return _generate_star_topology(num_nodes)
    elif topology_type == "complete":
        return _generate_complete_topology(num_nodes)
    elif topology_type == "small_world":
        return _generate_small_world_topology(num_nodes, connection_probability)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

def _generate_random_topology(num_nodes: int, prob: float) -> np.ndarray:
    """Generate random topology with given connection probability."""
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    return adjacency_matrix

def _generate_ring_topology(num_nodes: int) -> np.ndarray:
    """Generate ring topology where each node connects to its neighbors."""
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        adjacency_matrix[i, next_node] = 1
        adjacency_matrix[next_node, i] = 1
    return adjacency_matrix

def _generate_star_topology(num_nodes: int) -> np.ndarray:
    """Generate star topology with one central node."""
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(1, num_nodes):
        adjacency_matrix[0, i] = 1
        adjacency_matrix[i, 0] = 1
    return adjacency_matrix

def _generate_complete_topology(num_nodes: int) -> np.ndarray:
    """Generate complete topology where every node connects to every other."""
    adjacency_matrix = np.ones((num_nodes, num_nodes), dtype=int)
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
    return adjacency_matrix

def _generate_small_world_topology(num_nodes: int, rewire_prob: float) -> np.ndarray:
    """Generate small-world topology using Watts-Strogatz model."""
    # Start with ring lattice
    adjacency_matrix = _generate_ring_topology(num_nodes)
    
    # Add connections to next-nearest neighbors
    for i in range(num_nodes):
        next_next = (i + 2) % num_nodes
        adjacency_matrix[i, next_next] = 1
        adjacency_matrix[next_next, i] = 1
    
    # Rewire edges with given probability
    edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) 
             if adjacency_matrix[i, j] == 1]
    
    for i, j in edges:
        if np.random.random() < rewire_prob:
            # Remove old edge
            adjacency_matrix[i, j] = 0
            adjacency_matrix[j, i] = 0
            
            # Add new random edge
            new_target = np.random.choice([k for k in range(num_nodes) if k != i])
            adjacency_matrix[i, new_target] = 1
            adjacency_matrix[new_target, i] = 1
    
    return adjacency_matrix

def analyze_topology(adjacency_matrix: np.ndarray) -> dict:
    """Analyze network topology properties."""
    num_nodes = adjacency_matrix.shape[0]
    num_edges = np.sum(adjacency_matrix) // 2
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    
    # Degree distribution
    degrees = np.sum(adjacency_matrix, axis=1)
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": np.mean(degrees),
        "degree_distribution": degrees
    }

def visualize_topology(adjacency_matrix: np.ndarray, title: str = "Network Topology"):
    """Visualize the network topology using NetworkX and Matplotlib."""
    G = nx.from_numpy_array(adjacency_matrix)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=12, font_weight='bold')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    num_nodes = 8
    
    # Generate different topologies
    topologies = ["random", "ring", "star", "complete", "small_world"]
    
    for topology in topologies:
        print(f"\n{topology.upper()} TOPOLOGY:")
        network = generate_network_topology(num_nodes, topology, seed=42)
        print(network)
        
        # Analyze topology
        analysis = analyze_topology(network)
        print(f"Analysis: {analysis}")
        
        # Visualize (uncomment to display)
        # visualize_topology(network, f"{topology.capitalize()} Network")
