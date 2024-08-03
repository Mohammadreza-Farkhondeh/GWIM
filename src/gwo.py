import logging
import os
from multiprocessing import current_process
from typing import Callable, Optional, Tuple

import numpy as np

from src.network import Network


def configure_logger(network_name: str) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{network_name}_{current_process().name}.log")

    logger = logging.getLogger(network_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger


class Wolf:
    def __init__(
        self, network: Network, seed_set_size: int, logger: logging.Logger
    ) -> None:
        self.network = network
        self.seed_set_size = seed_set_size
        self.logger = logger
        self.position: np.ndarray = self.get_random_position()
        self.fitness: float = self.evaluate_fitness()
        self.logger.debug(
            f"Wolf initialized with position: {self.position} and fitness: {self.fitness}"
        )

    def get_random_position(self) -> np.ndarray:
        self.logger.debug("Getting random position")
        degrees = self.network.degrees_df[self.network.degrees_df["degree"] > 1][
            "degree"
        ].values
        max_degree = self.network.max_degree
        position = np.random.random(len(self.network.v_prime)) * degrees / max_degree
        self.logger.debug(f"Generated random position: {position}")
        return position

    def get_seed_set(self) -> np.ndarray:
        self.logger.debug("Computing seed set")
        node_prob_pairs = np.column_stack((self.network.v_prime, self.position))
        sorted_nodes = node_prob_pairs[node_prob_pairs[:, 1].argsort()[::-1]]
        seed_set = sorted_nodes[: self.seed_set_size, 0].astype(int)
        self.logger.debug(f"Computed seed set: {seed_set}")
        return seed_set

    def evaluate_fitness(self) -> float:
        self.logger.debug("Evaluating fitness")
        self.seed_set = self.get_seed_set()
        fitness = self.network.evaluate_fitness(self.seed_set)
        self.logger.info(f"Evaluated fitness: {fitness} for seed set: {self.seed_set}")
        return fitness

    def update_position(
        self,
        alpha: "Wolf",
        beta: "Wolf",
        delta: "Wolf",
        a: float,
        seedset_optimizer: Optional[Callable] = None,
    ) -> None:
        self.logger.debug(
            f"Updating position with alpha, beta, delta fitness: {alpha.fitness}, {beta.fitness}, {delta.fitness}"
        )
        A1, C1 = (
            2 * a * np.random.random(len(self.position)) - a,
            2 * np.random.random(len(self.position)),
        )
        A2, C2 = (
            2 * a * np.random.random(len(self.position)) - a,
            2 * np.random.random(len(self.position)),
        )
        A3, C3 = (
            2 * a * np.random.random(len(self.position)) - a,
            2 * np.random.random(len(self.position)),
        )

        D_alpha = abs(C1 * alpha.position - self.position)
        D_beta = abs(C2 * beta.position - self.position)
        D_delta = abs(C3 * delta.position - self.position)

        X1 = alpha.position - A1 * D_alpha
        X2 = beta.position - A2 * D_beta
        X3 = delta.position - A3 * D_delta

        self.position = (X1 + X2 + X3) / 3
        self.position = np.clip(self.position, 0, 1)
        self.fitness = self.evaluate_fitness()

        self.logger.debug(
            f"Updated position: {self.position} with fitness: {self.fitness}"
        )

        if seedset_optimizer is not None and callable(seedset_optimizer):
            new_seed_set = seedset_optimizer(self.network, self.seed_set)
            if not np.array_equal(new_seed_set, self.seed_set):
                self.seed_set = new_seed_set
                self.fitness = self.evaluate_fitness()
                self.logger.info(
                    f"Updated seed set to: {self.seed_set} with fitness: {self.fitness}"
                )


class GWIMOptimizer:
    def __init__(
        self,
        network: Network,
        population_size: int,
        seed_set_size: int,
        max_iter: int,
        seedset_optimizer: Optional[Callable] = None,
    ) -> None:
        self.logger = configure_logger(network.name)
        self.network = network
        self.population_size = population_size
        self.seed_set_size = seed_set_size
        self.max_iter = max_iter
        self.seedset_optimizer = seedset_optimizer
        self.logger.debug(
            f"Initializing GWIMOptimizer with population_size={population_size}, seed_set_size={seed_set_size}, max_iter={max_iter}"
        )
        self.population: list = [
            Wolf(network, seed_set_size, self.logger) for _ in range(population_size)
        ]
        self.alpha, self.beta, self.delta = self.get_leaders()
        self.logger.info(
            f"Initialized GWIMOptimizer with alpha seed set: {self.alpha.seed_set}, beta seed set: {self.beta.seed_set}, delta seed set: {self.delta.seed_set}"
        )

    def get_leaders(self) -> Tuple[Wolf, Wolf, Wolf]:
        self.logger.debug("Determining leaders from population")
        sorted_population = sorted(
            self.population, key=lambda wolf: wolf.fitness, reverse=True
        )
        alpha = sorted_population[0]
        beta = (
            sorted_population[1]
            if sorted_population[1] != alpha
            else Wolf(self.network, self.seed_set_size, self.logger)
        )
        delta = (
            sorted_population[2]
            if sorted_population[2] != alpha
            else Wolf(self.network, self.seed_set_size, self.logger)
        )
        self.logger.info(
            f"Leaders determined: alpha fitness: {alpha.fitness}, beta fitness: {beta.fitness}, delta fitness: {delta.fitness}"
        )
        return alpha, beta, delta
