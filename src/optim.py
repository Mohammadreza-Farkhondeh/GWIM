import numpy as np
import rustworkx as rx

from src.network import Network


def non_optimizer(network: Network, seed_set: np.ndarray) -> np.ndarray:
    return seed_set


def neighbor_based_optimizer(network: Network, seed_set: np.ndarray) -> np.ndarray:
    new_seed_set = seed_set.copy()
    for i in range(len(seed_set)):
        node = seed_set[i]
        neighbors = network.graph.neighbors(node)
        neighbor_degrees = {neighbor: len(network.graph.neighbors(neighbor)) for neighbor in neighbors}
        if neighbor_degrees:
            max_neighbor = max(neighbor_degrees, key=neighbor_degrees.get)
            new_seed_set[i] = max_neighbor
    return new_seed_set


def shortest_path_replacement_optimizer(network: Network, seed_set: np.ndarray) -> np.ndarray:
    new_seed_set = seed_set.copy()
    for i in range(len(seed_set)):
        node = seed_set[i]
        shortest_paths = rx.dijkstra_shortest_paths(network.graph, node)
        path_lengths = {target: len(path) for target, path in shortest_paths.items() if target in seed_set and target != node}
        if path_lengths:
            min_path_node = min(path_lengths, key=path_lengths.get)
            new_seed_set[i] = min_path_node
    return new_seed_set


