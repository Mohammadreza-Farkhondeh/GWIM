import networkx as nx
import numpy as np


class Network:
    def __init__(self, graph: nx.Graph = None) -> None:
        if graph is not None:
            self.graph = graph

        self.v_prime: list[int] = self.get_v_prime()
        self.v_prime_size = len(self.v_prime)

    def evaluate_fitness(self, seed_set):
        s_prime = self.get_s_prime(seed_set)
        worthiness = {
            node: self.calculate_worthiness(node, seed_set) for node in s_prime
        }
        total_worthiness = sum(worthiness.values())

        if total_worthiness == 0:
            return 0

        entropy = -sum(
            (worthiness[node] / total_worthiness)
            * np.log(worthiness[node] / total_worthiness + 1e-10)
            for node in worthiness
        )

        return entropy

    def calculate_worthiness(self, node, seed_set):
        propagation_probability = self.calculate_propagation_probability(node, seed_set)
        degree = self.get_degree(node)
        worthiness = propagation_probability * degree
        return worthiness

    def calculate_propagation_probability(self, node, seed_set):
        propagation_probability = 0

        # Calculate probability from direct neighbors
        for neighbor in seed_set:
            propagation_probability += self.get_propagation_probability(
                node, neighbor, seed_set
            )

        # Calculate probability from second-order neighbors
        for neighbor1 in seed_set:
            for neighbor2 in self.get_neighbors(neighbor1):
                if neighbor2 in seed_set:
                    propagation_probability += self.get_propagation_probability(
                        node, neighbor1, seed_set
                    ) * self.get_propagation_probability(neighbor1, neighbor2, seed_set)

        return propagation_probability

    def get_propagation_probability(self, node1, node2, seed_set):
        if node2 in self.get_neighbors(node1) and node2 in seed_set:
            return 1

        else:
            return 0

    def get_v_prime(self):
        return [node for node, degree in self.get_degree() if degree > 1]

    def get_s_prime(self, seed_set):
        s_prime = set()
        for node in seed_set:
            s_prime.update(self.get_neighbors(node))  # First-order neighbors
            for neighbor in self.get_neighbors(node):
                s_prime.update(self.get_neighbors(neighbor))  # Second-order neighbors
        s_prime -= set(seed_set)  # Exclude seed set nodes from s_prime
        return s_prime

    def get_neighbors(self, node):
        return self.graph.neighbors(node)

    def get_degree(self, node=None):
        return self.graph.degree(node) if node else self.graph.degree()
