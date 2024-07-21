import networkx as nx
import numpy as np

from src.network import Network


def no_optimization(network: Network, seed_set: np.ndarray) -> np.ndarray:
    return seed_set


def replace_with_higher_degree_neighbor(
    network: Network, seed_set: np.ndarray
) -> np.ndarray:
    new_seed_set = seed_set.copy()

    for i, node in enumerate(seed_set):
        neighbors = network.get_neighbors(node)
        node_degree = network.get_degree(node)

        for neighbor in neighbors:
            neighbor_degree = network.get_degree(neighbor)
            if neighbor_degree > node_degree:
                temp_seed_set = new_seed_set.copy()
                temp_seed_set[i] = neighbor
                if network.evaluate_fitness(temp_seed_set) > network.evaluate_fitness(
                    new_seed_set
                ):
                    new_seed_set[i] = neighbor

    return new_seed_set


def shortest_path_replacement(network: Network, seed_set: np.ndarray) -> np.ndarray:
    new_seed_set = seed_set.copy()
    path_nodes = set()

    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            path = nx.shortest_path(
                network.graph, source=seed_set[i], target=seed_set[j]
            )
            path_nodes.update(path)

    path_nodes -= set(seed_set)
    if not path_nodes:
        return new_seed_set

    highest_degree_node = max(path_nodes, key=lambda node: network.get_degree(node))
    lowest_fitness_node_index = np.argmin(
        [network.calculate_worthiness(node, seed_set) for node in new_seed_set]
    )

    temp_seed_set = new_seed_set.copy()
    temp_seed_set[lowest_fitness_node_index] = highest_degree_node

    if network.evaluate_fitness(temp_seed_set) > network.evaluate_fitness(new_seed_set):
        new_seed_set = temp_seed_set

    return new_seed_set
