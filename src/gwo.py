import numpy as np
from typing import Optional, Callable, Tuple, List
import logging

from src.network import Network


class Wolf:
    def __init__(self, network: Network, seed_set_size: int) -> None:
        self.network = network
        self.seed_set_size = seed_set_size
        self.position: np.ndarray = self.get_random_position()
        self.fitness: float = self.evaluate_fitness()

    def get_random_position(self) -> np.ndarray:
        degrees = self.network.degrees_df[self.network.degrees_df["degree"] > 1]["degree"].values
        max_degree = self.network.max_degree
        return np.random.random(len(self.network.v_prime)) * degrees / max_degree

    def get_seed_set(self) -> np.ndarray:
        node_prob_pairs = np.column_stack((self.network.v_prime, self.position))
        sorted_nodes = node_prob_pairs[node_prob_pairs[:, 1].argsort()[::-1]]
        return sorted_nodes[:self.seed_set_size, 0].astype(int)

    def evaluate_fitness(self) -> float:
        self.seed_set = self.get_seed_set()
        return self.network.evaluate_fitness(self.seed_set)

    def update_position(
        self,
        alpha: "Wolf",
        beta: "Wolf",
        delta: "Wolf",
        a: float,
        seedset_optimizer: Optional[Callable] = None,
    ) -> None:
        A1, C1 = (2 * a * np.random.random(len(self.position)) - a, 2 * np.random.random(len(self.position)))
        A2, C2 = (2 * a * np.random.random(len(self.position)) - a, 2 * np.random.random(len(self.position)))
        A3, C3 = (2 * a * np.random.random(len(self.position)) - a, 2 * np.random.random(len(self.position)))

        D_alpha = abs(C1 * alpha.position - self.position)
        D_beta = abs(C2 * beta.position - self.position)
        D_delta = abs(C3 * delta.position - self.position)

        X1 = alpha.position - A1 * D_alpha
        X2 = beta.position - A2 * D_beta
        X3 = delta.position - A3 * D_delta

        self.position = (X1 + X2 + X3) / 3
        self.position = np.clip(self.position, 0, 1)
        self.fitness = self.evaluate_fitness()

        if seedset_optimizer is not None and callable(seedset_optimizer):
            new_seed_set = seedset_optimizer(self.network, self.seed_set)
            if not np.array_equal(new_seed_set, self.seed_set):
                self.seed_set = new_seed_set
                self.fitness = self.evaluate_fitness()


class GWIMOptimizer:
    def __init__(
        self,
        network: Network,
        population_size: int,
        seed_set_size: int,
        max_iter: int,
        seedset_optimizer: Optional[callable] = None,
    ) -> None:
        self.network = network
        self.population_size = population_size
        self.seed_set_size = seed_set_size
        self.max_iter = max_iter
        self.seedset_optimizer = seedset_optimizer
        self.population: List[Wolf] = [Wolf(network, seed_set_size) for _ in range(population_size)]
        self.alpha, self.beta, self.delta = self.get_leaders()
        logging.debug(
            f"Initialized GWIMOptimizer with alpha: {self.alpha.seed_set}, beta: {self.beta.seed_set}, delta: {self.delta.seed_set}"
        )

    def get_leaders(self) -> Tuple[Wolf, Wolf, Wolf]:
        sorted_population = sorted(self.population, key=lambda wolf: wolf.fitness, reverse=True)
        alpha = sorted_population[0]
        beta = sorted_population[1] if sorted_population[1] != alpha else Wolf(self.network, self.seed_set_size)
        delta = sorted_population[2] if sorted_population[2] != alpha else Wolf(self.network, self.seed_set_size)
        return alpha, beta, delta
