import cProfile
import pstats
from itertools import combinations

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from src import optim
from src.gwo import run_gwo_with_optimizer
from src.network import Network


def get_network(n: str) -> Network:
    datasets = {
        # email Enron dataset with 360k edges
        # https://snap.stanford.edu/data/email-Enron.txt.gz
        "enron": "seed/enron.csv",
        # soc twitter follow with 700k edges
        # https://nrvis.com/download/data/soc/soc-twitter-follows.zip
        "soc-twitter-follows": "seed/soc-twitter-follows.csv",
        # soc linkedin dataset with 19M edges
        # https://nrvis.com/download/data/soc/soc-linkedin.zip
        "soc-linkedin": "seed/soc-linkedin.edges",
        # congress twitter dataset with 13k edges
        # https://snap.stanford.edu/data/congress_network.zip
        "congress": "seed/congress.edgelist",
        # hamsterster dataset with 17k edges
        # https://nrvis.com/download/data/soc/soc-hamsterster.zip
        "hamsterster": "seed/soc-hamsterster.edges",
        # pages food dataet with 2K edges
        # https://nrvis.com/download/data/soc/fb-pages-food.zip
        "food": "seed/fb-pages-foods.edges",
        # pretty good privacy dataset with 25K edges
        # https://nrvis.com/download/data/tech/tech-pgp.zip
        "pgp": "seed/tech-pgp.edges",
    }

    if n in datasets:
        try:
            edgelist_df = pd.read_csv(datasets[n], sep=" ")
            graph = nx.from_pandas_edgelist(edgelist_df)
            network = Network(graph=graph, name=n)
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
        graph = nx.erdos_renyi_graph(100, 0.025)
        network = Network(graph=graph)
    else:
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


def plot_comparison(results, imname):
    num_optimizers = len(results)
    fig, axs = plt.subplots(
        1, num_optimizers, figsize=(num_optimizers * 6, 6), sharey=True
    )

    for i, (label, result) in enumerate(results.items()):
        iterations = range(1, len(result[1]) + 1)
        axs[i].plot(iterations, result[0], label="Alpha Fitness")
        axs[i].set_xlabel("Time / Iteration Number")
        axs[i].set_title(f"Alpha Fitness vs. Time/Iteration for {label}")
        axs[i].grid(True)

        ax2 = axs[i].twiny()
        ax2.plot(result[1], result[0], alpha=0)
        ax2.set_xlim(result[1][0], result[1][-1])
        ax2.set_xlabel("Time")

    axs[0].set_ylabel("Alpha Fitness")
    plt.tight_layout()
    plt.savefig(f"{imname}.jpg")


def get_result(network_name, population_size, seed_set_size, max_iter):
    network: Network = get_network(n=network_name)

    result = {
        "No Optimization": run_gwo_with_optimizer(
            network,
            optim.no_optimization,
            population_size=population_size,
            seed_set_size=seed_set_size,
            max_iter=max_iter,
        ),
        "Higher Degree Neighbor": run_gwo_with_optimizer(
            network,
            optim.replace_with_higher_degree_neighbor,
            population_size=population_size,
            seed_set_size=seed_set_size,
            max_iter=max_iter,
        ),
        "Path High Worthiness": run_gwo_with_optimizer(
            network,
            optim.shortest_path_replacement,
            population_size=population_size,
            seed_set_size=seed_set_size,
            max_iter=max_iter,
        ),
    }

    plot_comparison(results=result, imname=f"{network.name}_results")


def profile_code(network, population_size, seed_set_size, max_iter):
    pr = cProfile.Profile()
    pr.enable()

    # Run the optimizer
    result = run_gwo_with_optimizer(
        network,
        optim.replace_with_higher_degree_neighbor,
        population_size,
        seed_set_size,
        max_iter,
    )

    pr.disable()
    ps = pstats.Stats(pr).sort_stats("cumtime")
    ps.print_stats()
