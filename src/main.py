import logging

from src.gwo import GWIMOptimizer
from src.network import Network
from src.utils import get_network

logging.basicConfig(
    level=logging.NOTSET, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Script started.")

try:
    n = "watts"
    logging.info(f"Attempting to get the {n} network...")
    network: Network = get_network(n=n)
    logging.info(f"Network obtained successfully: {network}")
except Exception as e:
    logging.error(f"Error getting network: {e}")
    exit(1)

n = 20  # Number of wolves in the population
k = 3  # Desired seed set size (number of nodes in the seed set)
max_t = 3  # Maximum number of iterations
optimize = True  # Optimize using the NeighborsDiversityFitnessMixin optimizer

logging.info(
    f"Initializing the optimizer with n={n}, k={k}, max_t={max_t}, optimize={optimize}"
)
optimizer = GWIMOptimizer(network, n, k)

logging.info("Starting the optimization algorithm...")
best_wolf = optimizer.run_gwo(max_t, optimize=optimize)
logging.info(f"Optimization complete. Best seed set: {best_wolf.seed_set}")
