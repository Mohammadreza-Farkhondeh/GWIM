import networkx as nx

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

    def optimize_seed_set(self, omega_wolf: "Wolf") -> list[int]:
        """
        Optimizes a seed set by incorporating neighbor diversity into the selection process.

        Args:
            omega_wolf (Wolf): An object representing the omega wolf (current seed set) in the optimization algorithm.

        Returns:
            list[int]: A new optimized seed set with enhanced diversity.
        """
        top_neighbors = self.get_top_neighbors(omega_wolf.seed_set)
        neighbors = [n for v in top_neighbors.values() for n in v]
        neighbors_scores = [
            (
                node,
                self.influence[node]
                * self.calculate_diversity(node, omega_wolf.seed_set),
            )
            for node in neighbors
        ]
        seed_set_scores = [
            (
                node,
                self.influence[node]
                * self.calculate_diversity(node, omega_wolf.seed_set),
            )
            for node in omega_wolf.seed_set
        ]

        neighbors_scores.sort(key=lambda x: x[1], reverse=True)
        seed_set_scores.sort(key=lambda x: x[1], reverse=True)

        seed_set_scores[-1] = (
            neighbors_scores[0]
            if neighbors_scores[0][1] / 2 > seed_set_scores[-1][1]
            else seed_set_scores[-1]
        )
        new_seed_set = [node for node, _ in seed_set_scores]
        return new_seed_set
