import logging
from typing import Callable, Optional, Tuple

import numpy as np

from src.network import Network


class Wolf:
    def __init__(self, network: Network, seed_set_size: int) -> None:
        self.network = network
        self.seed_set_size = seed_set_size
        self.position: np.ndarray = self.get_random_position()
        self.fitness: float = self.evaluate_fitness()

    def get_random_position(self) -> np.ndarray:
        return np.array(
            [
                np.random.random()
                * int(self.network.get_degree(j))
                / self.network.max_degree
                for j in self.network.v_prime
            ]
        )

    def get_seed_set(self) -> np.ndarray:
        node_prob_pairs = np.column_stack((self.network.v_prime, self.position))
        sorted_nodes = node_prob_pairs[node_prob_pairs[:, 1].argsort()[::-1]]
        return sorted_nodes[: self.seed_set_size, 0].astype(int)

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
        for i in range(len(self.position)):
            A1, C1 = 2 * a * np.random.random() - a, 2 * np.random.random()
            A2, C2 = 2 * a * np.random.random() - a, 2 * np.random.random()
            A3, C3 = 2 * a * np.random.random() - a, 2 * np.random.random()
            D_alpha = abs(C1 * alpha.position[i] - self.position[i])
            D_beta = abs(C2 * beta.position[i] - self.position[i])
            D_delta = abs(C3 * delta.position[i] - self.position[i])
            X1 = alpha.position[i] - A1 * D_alpha
            X2 = beta.position[i] - A2 * D_beta
            X3 = delta.position[i] - A3 * D_delta
            self.position[i] = (X1 + X2 + X3) / 3

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
        self.population: list = [
            Wolf(network, seed_set_size) for _ in range(population_size)
        ]
        self.alpha, self.beta, self.delta = self.get_leaders()
        logging.debug(
            f"Initialized GWIMOptimizer with alpha: {self.alpha.seed_set}, beta: {self.beta.seed_set}, delta: {self.delta.seed_set}"
        )

    def get_leaders(self) -> Tuple[Wolf, Wolf, Wolf]:
        sorted_population = sorted(
            self.population, key=lambda wolf: wolf.fitness, reverse=True
        )
        return sorted_population[0], sorted_population[1], sorted_population[2]

    def run_gwo(self) -> np.ndarray:
        a = 2.0
        for iter_num in range(self.max_iter):
            logging.debug(
                f"Iteration {iter_num}, alpha wolf: {self.alpha.seed_set}, fitness: {self.alpha.fitness:.3f}"
            )
            for wolf in self.population:
                wolf.update_position(
                    self.alpha, self.beta, self.delta, a, self.seedset_optimizer
                )
            self.alpha, self.beta, self.delta = self.get_leaders()
            logging.debug(
                f"New alpha wolf: {self.alpha.seed_set}, fitness: {self.alpha.fitness:.3f}"
            )
            a -= 2 / self.max_iter
        return self.alpha.seed_set
