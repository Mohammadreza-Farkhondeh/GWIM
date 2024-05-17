import logging

from src.gwo import GWIMOptimizer
from src.network import Network
from src.utils import get_network

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Script started.")

try:
    n = "soc-twitter-follows"
    logging.info(f"Attempting to get the {n} network...")
    network: Network = get_network(n=n)
    logging.info(f"Network obtained successfully: {network}")
except Exception as e:
    logging.error(f"Error getting network: {e}")
    exit(1)

n = 50  # Number of wolves in the population
k = 3  # Desired seed set size (number of nodes in the seed set)
max_t = 100  # Maximum number of iterations

logging.info(f"Initializing the optimizer with n={n}, k={k}, max_t={max_t}")
optimizer = GWIMOptimizer(network, n, k)

logging.info("Starting the optimization algorithm...")
best_seed_set = optimizer.run_gwo(max_t, optimize=False)
logging.info(f"Optimization complete. Best seed set: {best_seed_set.seed_set}")
