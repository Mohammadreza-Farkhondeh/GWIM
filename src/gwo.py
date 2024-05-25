import numpy as np


class Wolf:
    def __init__(self, network, seed_set_size):
        self.network = network
        self.seed_set_size = seed_set_size
        self.position = np.random.rand(len(network.graph.nodes()))
        self.seed_set = self.get_seed_set()
        self.fitness = self.evaluate_fitness()

    def get_seed_set(self):
        node_prob_pairs = zip(self.network.v_prime, self.position)
        sorted_nodes = sorted(node_prob_pairs, key=lambda x: x[1], reverse=True)
        return {node for node, prob in sorted_nodes[: self.seed_set_size]}

    def evaluate_fitness(self):
        self.seed_set = self.get_seed_set()
        return self.network.evaluate_fitness(self.seed_set)

    def update_position(self, alpha, beta, delta, a, seedset_optimizer=None):
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
            if new_seed_set != self.seed_set:
                self.seed_set = new_seed_set


class GWIMOptimizer:
    def __init__(
        self, network, population_size, seed_set_size, max_iter, seedset_optimizer=None
    ):
        self.network = network
        self.population_size = population_size
        self.seed_set_size = seed_set_size
        self.max_iter = max_iter
        self.seedset_optimizer = seedset_optimizer
        self.population = [Wolf(network, seed_set_size) for _ in range(population_size)]
        self.alpha, self.beta, self.delta = self.get_leaders()

    def get_leaders(self):
        sorted_population = sorted(
            self.population, key=lambda wolf: wolf.fitness, reverse=True
        )
        return sorted_population[0], sorted_population[1], sorted_population[2]

    def run_gwo(self):
        a = 2
        for _ in range(self.max_iter):
            print(
                f"iter {_}, alpha wolf {self.alpha.seed_set}, fitness: {self.alpha.fitness:.3f}"
            )
            for wolf in self.population:
                wolf.update_position(
                    self.alpha, self.beta, self.delta, a, self.seedset_optimizer
                )
            self.alpha, self.beta, self.delta = self.get_leaders()
            a -= 2 / self.max_iter
        return self.alpha.seed_set
