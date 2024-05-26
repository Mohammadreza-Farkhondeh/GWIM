from itertools import combinations

import networkx as nx
import pandas as pd

from src.network import Network


def get_network(n: str) -> Network:
    datasets = {
        # email Enron dataset with 36k edges
        # https://snap.stanford.edu/data/email-Enron.txt.gz
        "enron": "seed/enron.csv",
        # soc twitter follow with 700k edges
        # https://nrvis.com/download/data/soc/soc-twitter-follows.zip
        "soc-twitter-follows": "seed/soc-twitter-follows.csv",
        # soc linkedin dataset with 19M edges
        # https://nrvis.com/download/data/soc/soc-linkedin.zip
        "soc-linkedin": "seed/soc-linkedin.edges",
    }

    if n in datasets:
        try:
            edgelist_df = pd.read_csv(datasets[n], sep=" ")
            graph = nx.from_pandas_edgelist(edgelist_df)
            network = Network(graph=graph)
        except FileNotFoundError:
            print(f"The file for {n} dataset was not found.")
        except pd.errors.EmptyDataError:
            print(f"The file for {n} dataset is empty.")
        except KeyError as e:
            print(f"Key {e} did not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    elif n == "barbasi":
        graph = nx.barabasi_albert_graph(100, 5)
        network = Network(graph=graph)
    elif n == "watts":
        graph = nx.watts_strogatz_graph(100, 5, 0.33)
        network = Network(graph=graph)
    elif n == "erdos":
        graph = nx.erdos_renyi_graph(100, 0.05)
        network = Network(graph=graph)
    else:
        nodes = list(range(1, 17))
        edges = [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 8),
            (5, 6),
            (5, 9),
            (6, 7),
            (6, 10),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (8, 10),
            (9, 10),
            (10, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (13, 14),
            (14, 15),
            (14, 16),
            (15, 16),
        ]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return Network(graph=graph)

    return network


def test_all_seed_sets(network):
    nodes = network.v_prime
    all_combinations = list(combinations(nodes, 3))

    fitness_results = []

    for seed_set in all_combinations:
        fitness = network.evaluate_fitness(set(seed_set))
        fitness_results.append((seed_set, fitness))

    return fitness_results
