import cProfile
import logging
import pstats
import random
import time
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
import rustworkx as rx
import seaborn as sns
from matplotlib import pyplot as plt

from src import optim
from src.gwo import GWIMOptimizer, Wolf
from src.network import Network


def get_network(n: str) -> Network:
    datasets = {
        "enron": "seed/enron.csv",
        "soc-twitter-follows": "seed/soc-twitter-follows.csv",
        "soc-linkedin": "seed/soc-linkedin.edges",
        "congress": "seed/congress.edgelist",
        "hamsterster": "seed/soc-hamsterster.edges",
        "food": "seed/fb-pages-foods.edges",
        "pgp": "seed/tech-pgp.edges",
    }

    if n in datasets:
        try:
            edgelist_df = pd.read_csv(
                datasets[n],
                sep=" ",
                header=0,
                dtype=np.int64,
            )
            graph = rx.PyGraph()
            node_indices = {}
            for _, row in edgelist_df.iterrows():
                if row["source"] not in node_indices:
                    node_indices[row["source"]] = graph.add_node(row["source"])
                if row["target"] not in node_indices:
                    node_indices[row["target"]] = graph.add_node(row["target"])
                graph.add_edge(
                    node_indices[row["source"]], node_indices[row["target"]], 1.0
                )
            network = Network(graph=graph, name=n)
            return network
        except FileNotFoundError:
            print(f"The file for {n} dataset was not found.")
        except pd.errors.EmptyDataError:
            print(f"The file for {n} dataset is empty.")
        except KeyError as e:
            print(f"Key {e} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    elif n == "barbasi":
        graph = rx.generators.barabasi_albert_graph(100, 5)
        network = Network(graph=graph)
    elif n == "watts":
        graph = rx.generators.watts_strogatz_graph(100, 5, 0.33)
        network = Network(graph=graph)
    elif n == "erdos":
        graph = rx.generators.erdos_renyi_graph(100, 0.025)
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
        graph = rx.PyGraph()
        node_indices = {}
        for edge in edges:
            if edge[0] not in node_indices:
                node_indices[edge[0]] = graph.add_node(edge[0])
            if edge[1] not in node_indices:
                node_indices[edge[1]] = graph.add_node(edge[1])
            graph.add_edge(node_indices[edge[0]], node_indices[edge[1]], 1.0)
        network = Network(graph=graph)

    return network


def test_all_seed_sets(network):
    nodes = network.v_prime
    all_combinations = list(combinations(nodes, 3))

    fitness_results = []

    for seed_set in all_combinations:
        fitness = network.evaluate_fitness(set(seed_set))
        fitness_results.append((seed_set, fitness))

    return fitness_results


def plot_comparison_(results, imname):
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


def plot_comparison(results, imname):
    sns.set(style="whitegrid")
    num_optimizers = len(results)
    fig, axs = plt.subplots(
        1, num_optimizers, figsize=(num_optimizers * 6, 6), sharey=True
    )
    colors = sns.color_palette("husl", num_optimizers)
    for i, (label, result) in enumerate(results.items()):
        iterations = range(1, len(result[1]) + 1)
        axs[i].plot(
            iterations,
            result[0],
            label="Alpha Fitness",
            color=colors[i],
            linewidth=2,
        )
        axs[i].set_xlabel("Time / Iteration Number", fontsize=12)
        axs[i].set_title(
            f"Alpha Fitness vs. Time for {label}",
            fontsize=14,
            fontweight="bold",
        )
        axs[i].grid(True, linestyle="--", alpha=0.6)

        ax2 = axs[i].twiny()
        ax2.plot(result[1], result[0], alpha=0)
        ax2.set_xlim(result[1][0], result[1][-1])
        ax2.set_xlabel("Time", fontsize=12)
        axs[i].legend(loc="best")

    axs[0].set_ylabel("Alpha Fitness", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{imname}_{random.randint(a=100, b=999)}.jpg", dpi=1080)


def profile_code(network, population_size, seed_set_size, max_iter):
    pr = cProfile.Profile()
    pr.enable()

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


def run_gwo_with_optimizer(
    network: Network, optimizer: Callable, population_size, seed_set_size, max_iter
) -> list[list]:
    logging.info(
        f"Running GWO with seedset optimizer {optimizer.__name__} for network {network.name}."
    )
    gwim_optimizer = GWIMOptimizer(
        network=network,
        population_size=population_size,
        seed_set_size=seed_set_size,
        max_iter=max_iter,
        seedset_optimizer=optimizer,
    )

    alpha_fitness_over_time = []
    fitness_time = []
    start_time = time.time()
    a = 2.0
    for iter_num in range(gwim_optimizer.max_iter):
        logging.debug(
            f"Iteration {iter_num}, alpha wolf: {gwim_optimizer.alpha.seed_set}, fitness: {gwim_optimizer.alpha.fitness:.3f}"
        )
        for wolf in gwim_optimizer.population:
            wolf.update_position(
                gwim_optimizer.alpha,
                gwim_optimizer.beta,
                gwim_optimizer.delta,
                a,
                gwim_optimizer.seedset_optimizer,
            )
        gwim_optimizer.alpha, gwim_optimizer.beta, gwim_optimizer.delta = (
            gwim_optimizer.get_leaders()
        )

        if np.array_equal(gwim_optimizer.beta.position, gwim_optimizer.alpha.position):
            gwim_optimizer.beta = Wolf(
                gwim_optimizer.network, gwim_optimizer.seed_set_size
            )
        if np.array_equal(gwim_optimizer.delta.position, gwim_optimizer.alpha.position):
            gwim_optimizer.delta = Wolf(
                gwim_optimizer.network, gwim_optimizer.seed_set_size
            )

        gwim_optimizer.alpha, gwim_optimizer.beta, gwim_optimizer.delta = (
            gwim_optimizer.get_leaders()
        )
        alpha_fitness_over_time.append(gwim_optimizer.alpha.fitness)
        fitness_time.append(time.time() - start_time)
        a -= 2 / gwim_optimizer.max_iter
        logging.debug(
            f"New alpha wolf: {gwim_optimizer.alpha.seed_set}, fitness: {gwim_optimizer.alpha.fitness:.3f}"
        )

    return [alpha_fitness_over_time, fitness_time, gwim_optimizer.alpha]


def get_result(args):
    network_name, population_size, seed_set_size, max_iter = args
    network: Network = get_network(n=network_name)

    result = {
        "No Optimization": run_gwo_with_optimizer(
            network,
            optim.non_optimizer,
            population_size=population_size,
            seed_set_size=seed_set_size,
            max_iter=max_iter,
        ),
        "Higher Degree Neighbor": run_gwo_with_optimizer(
            network,
            optim.neighbor_based_optimizer,
            population_size=population_size,
            seed_set_size=seed_set_size,
            max_iter=max_iter,
        ),
        "Path High Worthiness": run_gwo_with_optimizer(
            network,
            optim.shortest_path_replacement_optimizer,
            population_size=population_size,
            seed_set_size=seed_set_size,
            max_iter=max_iter,
        ),
    }

    plot_comparison(results=result, imname=f"{network.name}_results")
    return result
