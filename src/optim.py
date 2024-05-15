from typing import Callable

import networkx as nx

from src.gwo import Wolf
from src.network import Network


class NeighborsDiversityFitnessMixin:
    """
    Mixin class that provides functionalities for optimizing seed sets in a social network context
    by considering neighbor diversity along with fitness and influence.
    """

    def __init__(self, network: Network, h: int) -> None:
        """
        Initializes the mixin with a Network object and a parameter `h`.

        Args:
            network (Network): The network object representing the social network.
            h (int): The number of top neighbors to consider for each node in the seed set.
        """
        self.network: Network = network
        self.influence: dict[int, float] = nx.eigenvector_centrality(network.graph)
        self.h: int = h

    def get_top_neighbors(self, seed_set: set[int]) -> dict[int, list[int]]:
        """
        Identifies the top `h` neighbors with the highest degree for each node in the seed set.

        Args:
            seed_set (set[int]): The set of nodes representing the current seed set.

        Returns:
            dict[int, list[int]]: A dictionary where keys are nodes in the seed set and values are
                                  lists of their top `h` neighbors (sorted by degree in descending order).
        """
        top_neighbors = {}
        for node in seed_set:
            neighbors = list(self.network.graph.neighbors(node))
            neighbors.sort(key=lambda x: self.network.graph.degree(x), reverse=True)
            top_neighbors[node] = (
                neighbors[: self.h] if len(neighbors) > self.h else neighbors
            )
        return top_neighbors

    def calculate_diversity(self, neighbor: int, seed_set: set[int]) -> float:
        """
        Calculates the diversity score between a potential neighbor and the existing seed set.

        Args:
            neighbor (int): The potential neighbor node.
            seed_set (set[int]): The set of nodes representing the current seed set.

        Returns:
            float: A diversity score between 0 and 1, indicating how different the neighbor's
                  connections are compared to the seed set's connections (higher diversity is better).
        """
        seed_set_neighbors = set(self.get_neighbors(seed_set))
        neighbor_neighbors = set(self.get_neighbors([neighbor]))
        return 1 - len(seed_set_neighbors.intersection(neighbor_neighbors)) / len(
            seed_set_neighbors.union(neighbor_neighbors)
        )

    def get_neighbors(self, node_list: list[int]) -> set[int]:
        """
        Returns a set of all neighbors for a given list of nodes.

        Args:
            node_list (list[int]): A list of node IDs.

        Returns:
            set[int]: A set containing all neighbors of the nodes in the list.
        """
        neighbors = set()
        for node in node_list:
            neighbors.update(self.network.graph.neighbors(node))
        return neighbors

    def optimize_seed_set(
        self, omega_wolf: Wolf, fitness_function: Callable[[int], float], a: float
    ) -> list[int]:
        """
        Optimizes a seed set by incorporating neighbor diversity into the selection process.

        Args:
            omega_wolf (object): An object representing the omega wolf (current seed set) in the optimization algorithm.
            fitness_function (callable[[int], float]): A function that calculates the fitness score of a node.
            a (float): A weight parameter between 0 and 1 to control the balance between fitness and diversity.

        Returns:
            list[int]: A new optimized seed set with enhanced diversity.
        """
        combined_scores = []
        for node, influence_score in self.influence.items():
            if node in omega_wolf.seed_set:
                combined_scores.append((node, fitness_function(node), influence_score))

        top_neighbors = self.get_top_neighbors(omega_wolf.seed_set)
        for neighbor_list in top_neighbors.values():
            for neighbor in neighbor_list:
                fitness = fitness_function(neighbor)
                diversity = self.calculate_diversity(neighbor, omega_wolf.seed_set)
                combined_score = (
                    a * fitness + (1 - a) * diversity + self.influence[neighbor]
                )
                combined_scores.append((neighbor, combined_score))

        combined_scores.sort(key=lambda x: x[1], reverse=True)

        new_seed_set = combined_scores[: self.k]
        new_seed_set = [node for node, _ in new_seed_set]

        return new_seed_set
