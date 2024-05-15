from src.gwo import GWIMOptimizer
from src.network import Network

nodes = [1, 2, 3, 4, 5]
edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]

network = Network(nodes=nodes, edges=edges)

n = 50  # Number of wolves in the population
k = 3  # Desired seed set size (number of nodes in the seed set)
max_t = 100  # Maximum number of iterations

optimizer = GWIMOptimizer(network, n, k)


best_seed_set = optimizer.run_gwo(max_t, optimize=False)
print(f"Best seedset: {best_seed_set.seed_set}")
