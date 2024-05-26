import logging

from src.gwo import GWIMOptimizer
from src.network import Network
from src.optim import optimize_seed_set_based_on_neighbors
from src.utils import get_network

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

try:
    n = "erdos"
    logging.info(f"Attempting to get the {n} network...")
    network: Network = get_network(n=n)
    logging.info(f"Network obtained successfully: {network}")
except Exception as e:
    logging.error(f"Error getting network: {e}")
    exit(1)

population_size = 15
seed_set_size = 3
max_iter = 10
optimize = False
optimizer_func = optimize_seed_set_based_on_neighbors if optimize else None

logging.info(
    f"Initializing the optimizer with population_size={population_size}, "
    f"seed_set_size={seed_set_size}, max_iter={max_iter} with optimizer {optimizer_func}"
)
optimizer = GWIMOptimizer(
    network=network,
    population_size=population_size,
    seed_set_size=seed_set_size,
    max_iter=max_iter,
    seedset_optimizer=optimizer_func,
)

logging.info("Starting the optimization algorithm...")
best_wolf = optimizer.run_gwo()
logging.info(f"Optimization complete. Best seed set: {best_wolf}")
