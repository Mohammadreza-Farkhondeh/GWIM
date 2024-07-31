import numpy as np
import pandas as pd
from typing import Optional, Union, List


class Network:
    def __init__(self, adjacency_matrix: pd.DataFrame, name: str = "") -> None:
        self.graph = adjacency_matrix
        self.name = name
        self.degrees_df = pd.DataFrame({
            "node": self.graph.index,
            "degree": self.graph.sum(axis=1)
        })
        self.v_prime = self.degrees_df[self.degrees_df["degree"] > 1]["node"].to_numpy(dtype=np.int64)
        self.v_prime_size = len(self.v_prime)
        self.max_degree = self.degrees_df["degree"].max()

    def evaluate_fitness(self, seed_set: np.ndarray) -> float:
        s_prime: np.ndarray = self.get_s_prime(seed_set)
        worthiness = np.array([self.calculate_worthiness(node, seed_set) for node in s_prime])
        total_worthiness: float = np.sum(worthiness)
        if total_worthiness == 0:
            return 0.0
        proportions = worthiness / total_worthiness
        entropy: float = -np.sum(proportions * np.log(proportions + 1e-10))
        return entropy

    def calculate_worthiness(self, node: int, seed_set: np.ndarray) -> float:
        propagation_probability: float = self.calculate_propagation_probability(node, seed_set)
        degree: int = self.get_degree(node)
        worthiness: float = propagation_probability * degree
        return worthiness

    def calculate_propagation_probability(self, node: int, seed_set: np.ndarray) -> float:
        neighbors = self.get_neighbors(node)
        direct_influence = np.isin(neighbors, seed_set).astype(float)
        second_order_neighbors = np.concatenate([self.get_neighbors(n) for n in neighbors])
        second_order_influence = np.isin(second_order_neighbors, seed_set).astype(float)
        return np.sum(direct_influence) + np.sum(second_order_influence)

    def get_v_prime(self) -> np.ndarray:
        return self.degrees_df[self.degrees_df["degree"] > 1]["node"].to_numpy()

    def get_s_prime(self, seed_set: np.ndarray) -> np.ndarray:
        s_prime = set()
        for node in seed_set:
            s_prime.update(self.get_neighbors(node))
            for neighbor in self.get_neighbors(node):
                s_prime.update(self.get_neighbors(neighbor))
        s_prime = np.array(list(s_prime - set(seed_set)))
        return s_prime

    def get_neighbors(self, node: int) -> np.ndarray:
        return self.graph.columns[self.graph.loc[node] > 0].to_numpy()

    def get_degree(self, node: Optional[Union[int, list]] = None):
        if isinstance(node, (int, np.int64)):
            return self.graph.loc[node].sum()
        elif isinstance(node, (list, np.ndarray)):
            return self.graph.loc[node].sum(axis=1).to_numpy()
        elif node is None:
            return self.graph.sum(axis=1).to_numpy()
