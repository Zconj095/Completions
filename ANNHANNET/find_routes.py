from simulate_node_failures import modified_topology
import numpy as np
from collections import deque, defaultdict
import heapq

def find_routes(source, destination, network_topology, max_routes=5, max_path_length=10):
    """
    Finds multiple possible routes from source to destination using various algorithms.

    Parameters:
    - source: int, the source node
    - destination: int, the destination node
    - network_topology: numpy array, the network topology adjacency matrix
    - max_routes: int, maximum number of alternative routes to find
    - max_path_length: int, maximum allowed path length

    Returns:
    - List of routes (as lists of node indices), sorted by path length
    """
    if source == destination:
        return [[source]]
    
    # Convert adjacency matrix to adjacency list for easier processing
    adj_list = _matrix_to_adjacency_list(network_topology)
    
    # Find shortest path first
    shortest_path = _dijkstra(adj_list, source, destination)
    if not shortest_path:
        return []
    
    # Find alternative paths using k-shortest paths algorithm
    all_routes = _find_k_shortest_paths(adj_list, source, destination, max_routes, max_path_length)
    
    return all_routes

def _matrix_to_adjacency_list(matrix):
    """Convert adjacency matrix to adjacency list."""
    adj_list = defaultdict(list)
    rows, cols = matrix.shape
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] > 0:  # Connection exists
                weight = matrix[i, j] if matrix[i, j] != 1 else 1
                adj_list[i].append((j, weight))
    
    return adj_list

def _dijkstra(adj_list, source, destination):
    """Find shortest path using Dijkstra's algorithm."""
    distances = defaultdict(lambda: float('inf'))
    distances[source] = 0
    previous = {}
    pq = [(0, source)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == destination:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = previous.get(current)
            return path[::-1]
        
        for neighbor, weight in adj_list[current]:
            if neighbor not in visited:
                new_dist = distances[current] + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return []

def _find_k_shortest_paths(adj_list, source, destination, k, max_length):
    """Find k shortest paths using Yen's algorithm (simplified version)."""
    if not adj_list or source not in adj_list:
        return []
    
    # Find shortest path
    shortest = _dijkstra(adj_list, source, destination)
    if not shortest:
        return []
    
    paths = [shortest]
    candidates = []
    
    for i in range(1, k):
        for j in range(len(paths[i-1]) - 1):
            # Create modified graph by removing edges
            spur_node = paths[i-1][j]
            root_path = paths[i-1][:j+1]
            
            # Remove edges that would create previously found paths
            modified_adj = defaultdict(list)
            for node in adj_list:
                modified_adj[node] = adj_list[node].copy()
            
            for path in paths:
                if len(path) > j and path[:j+1] == root_path:
                    if j+1 < len(path):
                        # Remove edge from spur_node to next node in path
                        next_node = path[j+1]
                        modified_adj[spur_node] = [
                            (n, w) for n, w in modified_adj[spur_node] 
                            if n != next_node
                        ]
            
            # Find spur path
            spur_path = _dijkstra(modified_adj, spur_node, destination)
            
            if spur_path and len(root_path[:-1] + spur_path) <= max_length:
                total_path = root_path[:-1] + spur_path
                if total_path not in paths and total_path not in [c[1] for c in candidates]:
                    path_cost = len(total_path) - 1  # Simple cost: number of hops
                    candidates.append((path_cost, total_path))
        
        if not candidates:
            break
            
        # Add shortest candidate to paths
        candidates.sort(key=lambda x: x[0])
        paths.append(candidates.pop(0)[1])
    
    return paths

def find_routes_bfs(source, destination, network_topology, max_routes=3):
    """
    Alternative route finding using BFS for simpler cases.
    
    Returns:
    - List of routes found via breadth-first search
    """
    if source == destination:
        return [[source]]
    
    adj_list = _matrix_to_adjacency_list(network_topology)
    routes = []
    queue = deque([(source, [source])])
    visited_paths = set()
    
    while queue and len(routes) < max_routes:
        current_node, path = queue.popleft()
        
        if len(path) > 6:  # Prevent overly long paths
            continue
            
        for neighbor, _ in adj_list[current_node]:
            if neighbor not in path:  # Avoid cycles
                new_path = path + [neighbor]
                path_tuple = tuple(new_path)
                
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    
                    if neighbor == destination:
                        routes.append(new_path)
                    else:
                        queue.append((neighbor, new_path))
    
    return sorted(routes, key=len)  # Sort by path length

# Example usage with enhanced functionality
if __name__ == "__main__":
    source = 0
    destination = 4
    
    # Find multiple routes using advanced algorithm
    routes = find_routes(source, destination, modified_topology, max_routes=3)
    print(f"Advanced routes from {source} to {destination}:")
    for i, route in enumerate(routes, 1):
        print(f"  Route {i}: {' -> '.join(map(str, route))} (length: {len(route)-1})")
    
    # Find routes using BFS for comparison
    bfs_routes = find_routes_bfs(source, destination, modified_topology)
    print(f"\nBFS routes from {source} to {destination}:")
    for i, route in enumerate(bfs_routes, 1):
        print(f"  Route {i}: {' -> '.join(map(str, route))} (length: {len(route)-1})")
