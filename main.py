import logging

from src import optim
from src.gwo import run_gwo_with_optimizer
from src.network import Network
from src.utils import get_network, plot_comparison

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

try:
    n = "hamsterstser"
    logging.info(f"Attempting to get the {n} network...")
    network: Network = get_network(n=n)
    logging.info(
        f"Network obtained successfully: number of edges {len(network.graph.edges())}"
    )
except Exception as e:
    logging.error(f"Error getting network: {e}")
    exit(1)

population_size = 10
seed_set_size = 4
max_iter = 25


results = {
    "No Optimization": run_gwo_with_optimizer(
        network,
        optim.no_optimization,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
    ),
    "Higher Degree Neighbor": run_gwo_with_optimizer(
        network,
        optim.replace_with_higher_degree_neighbor,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
    ),
    "High Degree": run_gwo_with_optimizer(
        network,
        optim.high_degree_nodes_optimizer,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
    ),
    "Random Perturbation": run_gwo_with_optimizer(
        network,
        optim.random_perturbation_optimizer,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
    ),
    "Greedy": run_gwo_with_optimizer(
        network,
        optim.greedy_influence_optimizer,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
    ),
    "Simulated Anealing": run_gwo_with_optimizer(
        network,
        optim.simulated_annealing_optimizer,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
    ),
}

plot_comparison(results)
