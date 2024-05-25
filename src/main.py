from src.gwo import GWIMOptimizer
from src.network import Network
from src.optim import optimize_seed_set_based_on_neighbors
from src.utils import get_network

print("Script started.")

try:
    n = "erdoss"
    print(f"Attempting to get the {n} network...")
    network: Network = get_network(n=n)
    print(f"Network obtained successfully: {network}")
except Exception as e:
    print(f"Error getting network: {e}")
    exit(1)

population_size = 5
seed_set_size = 5
max_iter = 3
optimizer = optimize_seed_set_based_on_neighbors

print(
    f"Initializing the optimizer with population_size={population_size}, seed_set_size={seed_set_size}, max_iter={max_iter} with optimzer {optimizer}"
)
optimizer = GWIMOptimizer(
    network=network,
    population_size=population_size,
    seed_set_size=5,
    max_iter=max_iter,
    seedset_optimizer=optimizer,
)

print("Starting the optimization algorithm...")
best_wolf = optimizer.run_gwo()

print(f"Optimization complete. Best seed set: {best_wolf}")
