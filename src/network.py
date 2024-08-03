from typing import Optional

import numpy as np
import pandas as pd
import rustworkx as rx


class Network:
    def __init__(self, graph: Optional[rx.PyGraph] = None, name: str = "") -> None:
        self.graph = graph if graph is not None else rx.PyGraph()
        self.name = name
        degrees = [
            (node, len(self.graph.neighbors(node)))
            for node in self.graph.node_indices()
        ]
        self.degrees_df = pd.DataFrame(degrees, columns=["node", "degree"])
        self.v_prime = self.degrees_df[self.degrees_df["degree"] > 1]["node"].to_numpy(
            dtype=np.int64
        )
        self.v_prime_size = len(self.v_prime)
        self.max_degree = self.degrees_df["degree"].max()

    def evaluate_fitness(self, seed_set: np.ndarray) -> float:
        s_prime = self.get_s_prime(seed_set)
        worthiness = np.array(
            [self.calculate_worthiness(node, seed_set) for node in s_prime]
        )
        total_worthiness = np.sum(worthiness)
        if total_worthiness == 0:
            return 0.0
        proportions = worthiness / total_worthiness
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        return entropy

    def calculate_worthiness(self, node: int, seed_set: np.ndarray) -> float:
        propagation_probability = self.calculate_propagation_probability(node, seed_set)
        degree = len(self.graph.neighbors(node))
        worthiness = propagation_probability * degree
        return worthiness

    def calculate_propagation_probability(
        self, node: int, seed_set: np.ndarray
    ) -> float:
        neighbors = self.graph.neighbors(node)
        direct_influence = np.isin(neighbors, seed_set).astype(float)
        second_order_neighbors = np.concatenate(
            [self.graph.neighbors(n) for n in neighbors]
        )
        second_order_influence = np.isin(second_order_neighbors, seed_set).astype(float)
        return np.sum(direct_influence) + np.sum(second_order_influence)

    def get_s_prime(self, seed_set: np.ndarray) -> np.ndarray:
        s_prime = set()
        for node in seed_set:
            s_prime.update(self.graph.neighbors(node))
            for neighbor in self.graph.neighbors(node):
                s_prime.update(self.graph.neighbors(neighbor))
        s_prime = np.array(list(s_prime - set(seed_set)))
        return s_prime
