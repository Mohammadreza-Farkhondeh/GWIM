import numpy as np

from src.network import Network


class Wolf:
    """
    Represents a candidate solution (wolf) in the GWIM algorithm.

    Attributes:
        v_prime (int): The effective network size (excluding isolated nodes).
        position (list[float]): List of probabilities for each node being a seed (length V').
        seed_set (list[Node]): List of the k most probable seed nodes.
        fitness_score (float): The calculated fitness score of the seed set.
    """

    def __init__(
        self,
        network: Network,
        k: int,
        seedset: list[int] = [],
        position: list[float] = [],
    ):
        """
        Initializes a Wolf object.

        Args:
            network (Network): The network object.
            k (int): The size of the seed set.
            seedset (list[int], optional): Initial seed set (default: []).
            position (list[float], optional): Initial position vector (default: []).
        """

        self.v_prime = network.get_v_prime()
        self.position = (
            position if position else [0.0] * self.v_prime
        )  # Initialize with zero probabilities

        self.seed_set = seedset
        self.update_seed_set(network, k) if seedset else None
        self.fitness_score: float = None

    def update_position(
        self, X_alpha: list, X_beta: list, X_delta: list, A: list, C: list
    ) -> None:
        """
        Updates the wolf's position (seed set probabilities) based on Alpha, Beta, and Delta wolves.

        Args:
            X_alpha (list): Position vector of the Alpha wolf.
            X_beta (list): Position vector of the Beta wolf.
            X_delta (list): Position vector of the Delta wolf.
            A (list): List containing A1, A2, A3 coefficients (length 3).
            C (list): List containing C1, C2, C3 coefficients (length 3).
        """

        new_position = []
        for i in range(len(self.position)):
            D_alpha = abs(C[0] * X_alpha[i] - self.position[i])
            Y_1 = X_alpha[i] - A[0] * D_alpha
            D_beta = abs(C[1] * X_beta[i] - self.position[i])
            Y_2 = X_beta[i] - A[1] * D_beta
            D_delta = abs(C[2] * X_delta[i] - self.position[i])
            Y_3 = X_delta[i] - A[2] * D_delta

            new_position.append((Y_1 + Y_2 + Y_3) / 3)

        self.position = new_position

    def update_seed_set(self, network: Network, k: int):
        """
        Updates the seed set based on the current position vector.

        Args:
            network (Network): The network object.
            k (int): The size of the seed set.
        """

        v_prime = self.v_prime
        sorted_nodes = sorted(
            network.get_nodes()[:v_prime], key=lambda x: self.position[x], reverse=True
        )
        self.seed_set = sorted_nodes[:k]

    def set_fitness_score(self, fitness_score: float):
        self.fitness_score = fitness_score


class GWIMOptimizer:
    """
    Implements the Grey Wolf Optimization (GWO) algorithm for identifying optimal seed sets
    in a social network diffusion process.

    Input:  Undirected Graph G=(V,E),
            the seed set size k,
            the population size n,
            the number of iterations max_t

    Output: by calling run_gwo ->
            S // a set with k members as initial seed set

    Attributes:
        population (list[Wolf]): List of wolf objects representing the GWO population.
        alpha_wolf (Wolf): Reference to the alpha wolf with the highest fitness.
        beta_wolf (Wolf): Reference to the beta wolf with the second-highest fitness.
        delta_wolf (Wolf): Reference to the delta wolf with the third-highest fitness.
        omega_wolves (list[Wolf]): List of wolf objects excluded from population, mostly playing the role of scapegoat.
        network (networkx.Graph or networkx.DiGraph): The NetworkX graph representing the social network.
    """

    def __init__(self, network: Network, n: int, k: int) -> None:
        """
        Initializes a GWIMOptimizer object.

        Args:
            network (networkx.Graph or networkx.DiGraph): The NetworkX graph representing the social network.
            n (int): The number of wolves in the GWO population.
            k (int): The size of seed sets for each wolf.
        """
        self.n = n
        self.k = k
        self.population = []
        self.alpha_wolf: Wolf = None
        self.beta_wolf: Wolf = None
        self.delta_wolf: Wolf = None
        self.omega_wolves: list[Wolf] = None
        self.network = network

        self.initialize_population(n, k)

    def generate_random_position(self, k: int) -> tuple[list, list]:
        """
        Generates a random position vector (X_i) and corresponding seed set (S_i)
        based on node degrees.

        Args:
            k (int): The size of the seed set.

        Returns:
            tuple[list, SeedSet]: A tuple containing the position vector (X_i) and seed set (S_i).
        """
        degrees = [self.network.graph.degree(node) for node in self.network.v_prime]
        max_degree = max(degrees)

        X_i = []
        for j in range(len(self.network.v_prime)):
            r = np.random.random() * degrees[j]
            X_i.append(r / max_degree)

        seed_indices = np.argsort(X_i)[::-1][:k]

        return X_i, seed_indices.tolist()

    def initialize_population(self, n: int, k: int) -> None:
        """
        Initializes the GWO population with random seed sets based on node degrees.

        Args:
            n (int): The number of wolves in the GWO population.
            k (int): The size of seed sets for each wolf.
        """

        for _ in range(n):
            X_i, seedset = self.generate_random_position(k)
            self.population.append(
                Wolf(network=self.network, seedset=seedset, k=k, position=X_i)
            )

        self.evaluate_population_fitness()

    def check_and_adjust_index(self, p1: int, p2: int, p3: int, index: int) -> int:
        """
        Ensures the updated node index stays within the valid range of the network's nodes.

        Args:
            p1 (int): Updated position value based on alpha wolf.
            p2 (int): Updated position value based on beta wolf.
            p3 (int): Updated position value based on delta wolf.
            index (int): The index of the node being updated in the seed set.

        Returns:
            int: The adjusted node index within the network's node range.
        """

        nodes = list(self.network.graph.nodes())
        new_value = int(round((p1 + p2 + p3) / 3))
        return min(max(new_value, 0), len(nodes) - 1)

    def get_linear_decay(self, iterations: int, max_iteration: int) -> float:
        """
        Calculates a linear decay value between 0 and 2 based on the current iteration.

        Args:
            iterations (int): The current iteration number.

        Returns:
            float: The linear decay value between 0 and 2.
        """
        # TODO: check for denominator, whether it should be
        #   - len(population) or
        #   - max iteration given in run_gwo
        return 2 * (1 - (iterations / (max_iteration)))

    def check_similar_wolves(self, wolf1: Wolf, wolf2: Wolf) -> bool:
        return wolf1.seed_set == wolf2.seed_set

    def fitness_function(self, seedset: list[int]) -> float:
        """
        Calculates the fitness score of a seed set using entropy-based influence spread estimation.

        Args:
            seedset (list[int]): List of node indices representing the seed set.

        Returns:
            float: The calculated fitness score of the seed set.
        """

        network = self.network
        W_S = 0  # Total worthiness of nodes in the seed set

        for node in seedset:
            # Calculate node worthiness (w(v_j))
            w_j = network.graph.degree(node) * len(
                list(network.graph.neighbors(node))
            )  # l(v_j) * d_j

            W_S += w_j

        # Calculate entropy (H(S))
        entropy = 0
        for node in seedset:
            w_j = network.graph.degree(node) * len(list(network.graph.neighbors(node)))
            p_j = w_j / W_S  # Proportion of worthiness for node j
            entropy -= p_j * np.log(p_j)

        return entropy

    def update_wolf_hierarchy(self) -> None:
        """
        Identifies the alpha, beta, and delta wolves based on their fitness scores.
        """

        self.population.sort(key=lambda w: w.fitness_score, reverse=True)
        self.alpha_wolf = self.population[0]
        self.beta_wolf = self.population[1]
        self.delta_wolf = self.population[2]
        self.omega_wolves = self.population[3:]

    def evaluate_population_fitness(self) -> None:
        """
        Calculates the fitness score of each wolf in the population using the fitness_function.
        """

        for wolf in self.population:
            fitness_score = self.fitness_function(wolf.seedset)
            wolf.set_fitness_score(fitness_score)

    def run_gwo(self, max_t: int) -> Wolf:
        """
        Runs the GWIM algorithm for influence maximization.

        Args:
            max_t (int): The maximum number of iterations.

        Returns:
            SeedSet: The final seed set with the highest predicted influence spread.
        """

        a = self.get_linear_decay(0, max_t)
        A = [a * (2 * np.random.random() - 1) for _ in range(3)]
        C = [2 * np.random.random() for _ in range(3)]

        self.evaluate_population_fitness()
        self.update_wolf_hierarchy()

        for t in range(max_t):
            # Omega wolves update
            for omega_wolf in self.omega_wolves:
                omega_wolf.update_position(
                    X_alpha=self.alpha_wolf.position,
                    X_beta=self.beta_wolf.position,
                    X_delta=self.delta_wolf.position,
                    A=A,
                    C=C,
                )
                omega_wolf.update_seed_set(network=self.network, k=self.k)

            self.evaluate_population_fitness()
            self.update_wolf_hierarchy()

            a = self.get_linear_decay(0, max_t)
            A = [a * (2 * np.random.random() - 1) for _ in range(3)]
            C = [2 * np.random.random() for _ in range(3)]

            if self.check_similar_wolves(
                self.alpha_wolf, self.beta_wolf
            ) or self.check_similar_wolves(self.beta_wolf, self.delta_wolf):
                X, S = self.generate_random_position(k=self.k)
                self.beta_wolf = Wolf(
                    network=self.network, position=X, seedset=S, k=self.k
                )
                X, S = self.generate_random_position(k=self.k)
                self.delta_wolf = Wolf(
                    network=self.network, position=X, seedset=S, k=self.k
                )

        return self.alpha_wolf
