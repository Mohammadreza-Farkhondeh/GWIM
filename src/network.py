import networkx as nx
import numpy as np

np.seterr(all="ignore")


class Network:
    def __init__(
        self,
        graph: nx.Graph = None,
    ) -> None:
        if graph is not None:
            self.graph = graph

        self.v_prime: list[int] = self.get_v_prime()
        self.v_prime_size = len(self.v_prime)

    def evaluate_fitness(self, seed_set):
        worthiness = {
            node: self.calculate_worthiness(node, seed_set) for node in seed_set
        }
        total_worthiness = sum(worthiness.values())
        if total_worthiness == 0:
            return 0
        entropy = -sum(
            (worthiness[node] / total_worthiness)
            * np.log(worthiness[node] / total_worthiness)
            for node in worthiness
        )
        return entropy

    def calculate_worthiness(self, node, seed_set):
        propagation_probability = self.calculate_propagation_probability(node, seed_set)
        degree = self.graph.degree(node)
        return propagation_probability * degree

    def calculate_propagation_probability(self, node, seed_set):
        neighbors_in_seed = set(self.graph.neighbors(node)) & seed_set
        second_order_neighbors = (
            set(
                neigh
                for neighbor in neighbors_in_seed
                for neigh in self.graph.neighbors(neighbor)
            )
            - seed_set
        )

        propagation_probability = 0

        # Calculate probability from direct neighbors
        for neighbor in neighbors_in_seed:
            propagation_probability += 1

        # Calculate probability from second-order neighbors
        for neigh1 in second_order_neighbors:
            for neigh2 in set(self.graph.neighbors(neigh1)) & seed_set:
                propagation_probability += 1

        return propagation_probability

    def get_v_prime(self):
        return [node for node, degree in self.graph.degree if degree > 1]

    def get_neighbors(self, node):
        return self.graph.neighbors(node)
