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

        max_degree_neighbor = max(neighbors, key=lambda n: network.get_degree(n))

        if node_degree > network.get_degree(max_degree_neighbor):
            new_seed_set[i] = max_degree_neighbor

    return new_seed_set


def fit_in_higher_degree_neighbor(network: Network, seed_set: np.ndarray) -> np.ndarray:
    optimized_seed_set = seed_set.copy()

    for i, node in enumerate(seed_set):
        neighbors = network.get_neighbors(node)

        best_neighbor = node
        best_influence = network.evaluate_fitness(optimized_seed_set)

        for neighbor in neighbors:
            if neighbor not in optimized_seed_set:
                temp_seed_set = optimized_seed_set.copy()
                temp_seed_set[i] = neighbor

                temp_influence = network.evaluate_fitness(temp_seed_set)

                if temp_influence > best_influence:
                    best_neighbor = neighbor
                    best_influence = temp_influence

        optimized_seed_set[i] = best_neighbor

    return optimized_seed_set


def diverse_higher_central_neighbor(
    network: Network, seed_set: np.ndarray, diversity_factor=0.1
) -> np.ndarray:
    optimized_seed_set = seed_set.copy()

    for i, node in enumerate(seed_set):
        neighbors = network.get_neighbors(node)
        best_neighbor = node
        best_heuristic = combined_heuristic(network, node)

        for neighbor in neighbors:
            if neighbor not in optimized_seed_set:
                heuristic_value = combined_heuristic(network, neighbor)

                if (
                    heuristic_value > best_heuristic
                    or np.random.random() < diversity_factor
                ):
                    best_neighbor = neighbor
                    best_heuristic = heuristic_value

        optimized_seed_set[i] = best_neighbor

    return optimized_seed_set


def combined_heuristic(network: Network, node: int) -> float:
    degree = network.get_degree(node)
    clustering_coefficient = nx.clustering(network.graph, node)
    return degree + clustering_coefficient


#######################################################################


def random_perturbation_optimizer(
    network: Network, seed_set: np.ndarray, perturbation_rate: float = 0.1
) -> np.ndarray:
    new_seed_set = seed_set.copy()
    num_perturbations = int(len(seed_set) * perturbation_rate)
    all_nodes = set(network.v_prime)

    for _ in range(num_perturbations):
        node_to_remove = np.random.choice(new_seed_set)
        new_seed_set = new_seed_set[new_seed_set != node_to_remove]
        remaining_nodes = list(all_nodes - set(new_seed_set))
        node_to_add = np.random.choice(remaining_nodes)
        new_seed_set = np.append(new_seed_set, node_to_add)

    return new_seed_set


def high_degree_nodes_optimizer(
    network: Network, seed_set: np.ndarray, replace_rate: float = 0.1
) -> np.ndarray:
    new_seed_set = seed_set.copy()
    num_replacements = int(len(seed_set) * replace_rate)
    node_degrees = {node: network.get_degree(node) for node in network.v_prime}
    sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

    for i in range(num_replacements):
        node_to_remove = np.random.choice(new_seed_set)
        new_seed_set = new_seed_set[new_seed_set != node_to_remove]
        for high_degree_node in sorted_nodes:
            if high_degree_node not in new_seed_set:
                new_seed_set = np.append(new_seed_set, high_degree_node)
                break

    return new_seed_set


def greedy_influence_optimizer(network: Network, seed_set: np.ndarray) -> np.ndarray:
    best_seed_set = seed_set.copy()
    best_fitness = network.evaluate_fitness(best_seed_set)

    for node in network.v_prime:
        if node not in best_seed_set:
            for seed in best_seed_set:
                new_seed_set = best_seed_set.copy()
                new_seed_set = new_seed_set[new_seed_set != seed]
                new_seed_set = np.append(new_seed_set, node)
                new_fitness = network.evaluate_fitness(new_seed_set)
                if new_fitness > best_fitness:
                    best_seed_set = new_seed_set
                    best_fitness = new_fitness

    return best_seed_set


def simulated_annealing_optimizer(
    network: Network,
    seed_set: np.ndarray,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.99,
    min_temp: float = 0.01,
) -> np.ndarray:
    current_seed_set = seed_set.copy()
    current_fitness = network.evaluate_fitness(current_seed_set)
    best_seed_set = current_seed_set.copy()
    best_fitness = current_fitness
    temp = initial_temp

    while temp > min_temp:
        new_seed_set = current_seed_set.copy()
        node_to_remove = np.random.choice(new_seed_set)
        new_seed_set = new_seed_set[new_seed_set != node_to_remove]
        remaining_nodes = list(set(network.v_prime) - set(new_seed_set))
        node_to_add = np.random.choice(remaining_nodes)
        new_seed_set = np.append(new_seed_set, node_to_add)
        new_fitness = network.evaluate_fitness(new_seed_set)

        if new_fitness > current_fitness or np.random.random() < np.exp(
            (new_fitness - current_fitness) / temp
        ):
            current_seed_set = new_seed_set
            current_fitness = new_fitness
            if new_fitness > best_fitness:
                best_seed_set = new_seed_set
                best_fitness = new_fitness

        temp *= cooling_rate

    return best_seed_set
