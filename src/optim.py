import numpy as np
import pandas as pd

from src.network import Network


def no_optimization(network: Network, seed_set: np.ndarray) -> np.ndarray:
    return seed_set

def replace_with_higher_degree_neighbor(network: Network, seed_set: np.ndarray) -> np.ndarray:
    new_seed_set = seed_set.copy()
    current_fitness = network.evaluate_fitness(new_seed_set)

    for i, node in enumerate(seed_set):
        neighbors = network.get_neighbors(node)
        node_degree = network.get_degree(node)

        for neighbor in neighbors:
            neighbor_degree = network.get_degree(neighbor)
            if neighbor_degree > node_degree:
                temp_seed_set = new_seed_set.copy()
                temp_seed_set[i] = neighbor
                new_fitness = network.evaluate_fitness(temp_seed_set)
                if new_fitness > current_fitness:
                    new_seed_set[i] = neighbor
                    current_fitness = new_fitness
    return new_seed_set

def shortest_path_replacement(network: Network, seed_set: np.ndarray) -> np.ndarray:
    new_seed_set = seed_set.copy()
    path_nodes = set()

    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            path = find_shortest_path(network.graph, seed_set[i], seed_set[j])
            path_nodes.update(path)

    path_nodes -= set(seed_set)
    if not path_nodes:
        return new_seed_set

    highest_degree_node = max(path_nodes, key=network.get_degree)
    lowest_fitness_node_index = np.argmin([network.calculate_worthiness(node, seed_set) for node in new_seed_set])

    temp_seed_set = new_seed_set.copy()
    temp_seed_set[lowest_fitness_node_index] = highest_degree_node

    current_fitness = network.evaluate_fitness(new_seed_set)
    new_fitness = network.evaluate_fitness(temp_seed_set)

    if new_fitness > current_fitness:
        new_seed_set = temp_seed_set

    return new_seed_set

def find_shortest_path(graph: pd.DataFrame, start_node: int, end_node: int) -> list:
    # Implement BFS or Dijkstra's algorithm for shortest path
    from collections import deque

    visited = {start_node}
    queue = deque([[start_node]])
    if start_node == end_node:
        return [start_node]

    while queue:
        path = queue.popleft()
        node = path[-1]

        for neighbor in graph.columns[graph.loc[node] > 0]:
            if neighbor == end_node:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return []
