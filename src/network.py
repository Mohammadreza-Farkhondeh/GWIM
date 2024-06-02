from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd


class Network:
    def __init__(self, graph: Optional[nx.Graph] = None, name: str = "") -> None:
        self.graph = graph if graph is not None else nx.Graph()
        self.name = name
        self.degrees_df = pd.DataFrame(self.graph.degree(), columns=["node", "degree"])
        self.v_prime = self.degrees_df[self.degrees_df["degree"] > 1]["node"].to_numpy(
            dtype=np.int64
        )
        self.v_prime_size = len(self.v_prime)
        self.max_degree = self.degrees_df["degree"].max()

    def evaluate_fitness(self, seed_set: np.ndarray) -> float:
        s_prime: np.ndarray = self.get_s_prime(seed_set)

        worthiness = np.array(
            [self.calculate_worthiness(node, seed_set) for node in s_prime]
        )
        total_worthiness: float = np.sum(worthiness)

        if total_worthiness == 0:
            return 0.0

        proportions = worthiness / total_worthiness
        entropy: float = -np.sum(proportions * np.log(proportions + 1e-10))
        return entropy

    def calculate_worthiness(self, node: int, seed_set: np.ndarray) -> float:
        propagation_probability: float = self.calculate_propagation_probability(
            node, seed_set
        )
        degree: int = self.get_degree(node)
        worthiness: float = propagation_probability * degree
        return worthiness

    def calculate_propagation_probability(
        self, node: int, seed_set: np.ndarray
    ) -> float:
        propagation_probability: float = 0.0
        for neighbor in seed_set:
            propagation_probability += self.get_propagation_probability(
                node, neighbor, seed_set
            )
        for neighbor1 in seed_set:
            for neighbor2 in self.get_neighbors(neighbor1):
                if neighbor2 in seed_set:
                    propagation_probability += self.get_propagation_probability(
                        node, neighbor1, seed_set
                    ) * self.get_propagation_probability(neighbor1, neighbor2, seed_set)

        return propagation_probability

    def get_propagation_probability(
        self, node1: int, node2: int, seed_set: np.ndarray
    ) -> float:
        if node2 in self.get_neighbors(node1) and node2 in seed_set:
            return 1.0
        return 0.0

    def get_v_prime(self) -> np.ndarray:
        return np.array([node for node, degree in self.get_degree() if degree > 1])

    def get_s_prime(self, seed_set: np.ndarray) -> np.ndarray:
        s_prime = set()

        for node in seed_set:
            s_prime.update(self.get_neighbors(node))  # First-order neighbors
            for neighbor in self.get_neighbors(node):
                s_prime.update(self.get_neighbors(neighbor))  # Second-order neighbors

        s_prime = np.array(list(s_prime - set(seed_set)))
        return s_prime

    def get_neighbors(self, node: int) -> np.ndarray:
        return np.array(list(self.graph.neighbors(node)))

    def get_degree(self, node: Optional[Union[int, list]] = None):
        if isinstance(node, int) or isinstance(node, np.int64):
            return self.degrees_df[self.degrees_df["node"] == node]["degree"].values[0]
        elif isinstance(node, list) or isinstance(node, np.ndarray):
            return self.degrees_df[self.degrees_df["node"].isin(node)][
                "degree"
            ].to_numpy()
        elif node is None:
            return self.degrees_df["degree"].to_numpy()

    def get_degree_values(self) -> np.ndarray:
        return np.array([degree for _, degree in self.get_degree()])
