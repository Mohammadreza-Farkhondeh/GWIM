import network as nx
import numpy as np

from src.network import Network


def optimize_seedset_degree(network: Network, seed_set: np.ndarray) -> np.ndarray:
    new_seed_set = seed_set.copy()

    for i, node in enumerate(seed_set):
        neighbors = network.get_neighbors(node)
        best_neighbor = node
        best_degree = network.get_degree(node)

        for neighbor in neighbors:
            neighbor_degree = network.get_degree(neighbor)
            if neighbor_degree > best_degree and neighbor not in seed_set:
                best_neighbor = neighbor
                best_degree = neighbor_degree

        new_seed_set[i] = best_neighbor

    return new_seed_set


def optimize_seedset_diverse_degree(network, current_seed_set, diversity_factor=0.1):
    optimized_seed_set = current_seed_set.copy()

    for node in current_seed_set:
        neighbors = network.get_neighbors(node)

        for neighbor in neighbors:
            if neighbor not in optimized_seed_set:
                temp_seed_set = optimized_seed_set.copy()
                temp_seed_set.remove(node)
                temp_seed_set.add(neighbor)

                temp_influence = network.evaluate_fitness(temp_seed_set)
                current_influence = network.evaluate_fitness(optimized_seed_set)

                if temp_influence > current_influence:
                    optimized_seed_set.remove(node)
                    optimized_seed_set.add(neighbor)
                    break
                else:
                    # temp_influence == current_influence
                    # and np.random.random() < diversity_factor
                    # ):
                    pass

    return optimized_seed_set


def combined_heuristic(network: Network, node: int) -> float:
    degree = network.get_degree(node)
    clustering_coefficient = nx.clustering(network.graph, node)
    return degree + clustering_coefficient
