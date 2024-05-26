import logging
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.gwo import GWIMOptimizer
from src.network import Network
from src.optim import optimize_seed_set_based_on_neighbors
from src.utils import get_network

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

try:
    n = "soc-twitter-follows"
    logging.info(f"Attempting to get the {n} network...")
    network: Network = get_network(n=n)
    logging.info(
        f"Network obtained successfully: number of edges {len(network.graph.edges())}"
    )
except Exception as e:
    logging.error(f"Error getting network: {e}")
    exit(1)

population_size = 20
seed_set_size = 3
max_iter = 25
optimize = False
seedset_optimizer_func = optimize_seed_set_based_on_neighbors if optimize else None

logging.info(
    f"Initializing the optimizer with population_size={population_size}, "
    f"seed_set_size={seed_set_size}, max_iter={max_iter} with seedset optimizer {seedset_optimizer_func}"
)
optimizer = GWIMOptimizer(
    network=network,
    population_size=population_size,
    seed_set_size=seed_set_size,
    max_iter=max_iter,
    seedset_optimizer=seedset_optimizer_func,
)

logging.info("Starting the optimization algorithm...")
best_wolf = optimizer.run_gwo()
logging.info(f"Optimization complete. Best seed set: {best_wolf}")


def evaluate_seedset_optimizer_effect(
    networks: List[nx.Graph],
    population_size: int,
    seed_set_size: int,
    max_iter: int,
    seedset_optimizer: Optional[Callable] = None,
) -> None:
    fitness_without_optimizer = []
    fitness_with_optimizer = []

    for graph in networks:
        network = Network(graph)

        optimizer_without = GWIMOptimizer(
            network, population_size, seed_set_size, max_iter
        )
        best_seed_set_without = optimizer_without.run_gwo()
        best_fitness_without = optimizer_without.alpha.fitness
        fitness_without_optimizer.append(best_fitness_without)

        optimizer_with = GWIMOptimizer(
            network, population_size, seed_set_size, max_iter, seedset_optimizer
        )
        best_seed_set_with = optimizer_with.run_gwo()
        best_fitness_with = optimizer_with.alpha.fitness
        fitness_with_optimizer.append(best_fitness_with)

    labels = [f"Network {i+1}" for i in range(len(networks))]
    x = np.arange(len(labels))

    width = 0.35
    fig, ax = plt.subplots()

    bars1 = ax.bar(
        x - width / 2, fitness_without_optimizer, width, label="Without Optimizer"
    )
    bars2 = ax.bar(x + width / 2, fitness_with_optimizer, width, label="With Optimizer")

    ax.set_xlabel("Networks")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness with and without Seedset Optimizer")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)

    fig.tight_layout()
    plt.show()


G1 = nx.erdos_renyi_graph(100, 0.05)
G2 = nx.barabasi_albert_graph(100, 5)
G3 = nx.watts_strogatz_graph(100, 6, 0.1)

networks = [G1, G2, G3]


evaluate_seedset_optimizer_effect(
    networks,
    population_size=10,
    seed_set_size=5,
    max_iter=50,
    seedset_optimizer=seedset_optimizer_func,
)
