import logging

from src import optim
from src.gwo import run_gwo_with_optimizer
from src.network import Network
from src.utils import get_network, plot_comparison

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

ns = ["food", "hamsterster", "soc-twitter-follows"]
networks = []

for n in ns:
    try:
        logging.info(f"Attempting to get the {n} network...")
        network: Network = get_network(n=n)
        logging.info(
            f"Network obtained successfully: number of edges {len(network.graph.edges())}"
        )
        networks.append(network)
    except Exception as e:
        logging.error(f"Error getting network: {e}")
        exit(1)

population_size = 100
seed_set_size = 5
max_iter = 50


results = [
    {
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
        # "Higher Degree Neighbor + Fitness": run_gwo_with_optimizer(
        #     network,
        #     optim.fit_in_higher_degree_neighbor,
        #     population_size=population_size,
        #     seed_set_size=seed_set_size,
        #     max_iter=max_iter,
        # ),
        # "Higher diverse neighbor heuristic": run_gwo_with_optimizer(
        #     network,
        #     optim.diverse_higher_central_neighbor,
        #     population_size=population_size,
        #     seed_set_size=seed_set_size,
        #     max_iter=max_iter,
        # ),
    }
    for network in networks
]

[
    plot_comparison(results=results, imname=f"{network.name}_results")
    for network in networks
]
