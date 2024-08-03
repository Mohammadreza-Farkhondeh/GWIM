import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.erdos_renyi_graph(20, 0.1)
seed_set = [3, 6, 9]
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)


def save_highlight_seed_neighbors(G, seed_set, filename):
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 8))

    neighbors = set()
    for seed in seed_set:
        neighbors.update(G.neighbors(seed))
    neighbors -= set(seed_set)

    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500)
    nx.draw_networkx_nodes(
        G, pos, nodelist=seed_set, node_color="red", node_size=700, label="Seed Nodes"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=neighbors,
        node_color="orange",
        node_size=500,
        label="Neighbors",
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title("Seed Set and Their Neighbors")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename), dpi=720)
    plt.close()


def save_highlight_shortest_paths(G, seed_set, filename):
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 8))

    path_nodes = set()
    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            try:
                path = nx.shortest_path(G, source=seed_set[i], target=seed_set[j])
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue

    path_nodes -= set(seed_set)

    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500)
    nx.draw_networkx_nodes(
        G, pos, nodelist=seed_set, node_color="red", node_size=700, label="Seed Nodes"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=path_nodes,
        node_color="blue",
        node_size=500,
        label="Path Nodes",
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title("Nodes in Shortest Paths Between Seed Nodes")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename), dpi=720)
    plt.close()


def save_heatmap_node_influence(G, seed_set, filename):
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))

    influence_scores = {node: G.degree(node) for node in G.nodes()}
    max_influence = max(influence_scores.values(), default=1)
    norm_influence_scores = {
        node: influence_scores[node] / max_influence for node in G.nodes()
    }

    node_colors = [
        plt.cm.viridis(norm_influence_scores.get(node, 0)) for node in G.nodes()
    ]
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        edge_color="gray",
        cmap=plt.cm.viridis,
        ax=ax,
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_influence)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Influence Score")

    plt.title("Heatmap of Node Influence")
    plt.savefig(os.path.join(output_dir, filename), dpi=720)
    plt.close()


def save_replacement_process_visualization(G, seed_set, filename):
    pos = nx.kamada_kawai_layout(G)

    for i, node in enumerate(seed_set):
        plt.figure(figsize=(10, 8))

        neighbors = list(G.neighbors(node))
        neighbor_degrees = {neighbor: G.degree(neighbor) for neighbor in neighbors}

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=seed_set,
            node_color="red",
            node_size=700,
            label="Seed Nodes",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=neighbors,
            node_color="orange",
            node_size=500,
            label="Neighbors",
        )

        if neighbor_degrees:
            highest_degree_neighbor = max(neighbor_degrees, key=neighbor_degrees.get)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[highest_degree_neighbor],
                node_color="blue",
                node_size=900,
                label="Replacement Node",
            )

        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos)

        plt.title(f"Replacement Process Step {i + 1}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{filename}_{i + 1}.png"), dpi=720)
        plt.close()


def save_influence_convergence_visualization(
    G, seed_set, iterations=4, filename="convergence.png"
):
    pos = nx.kamada_kawai_layout(G)
    cols = 2
    rows = (iterations // cols) + (1 if iterations % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 16, rows * 16))

    for iteration in range(iterations):
        active_nodes = set(seed_set)
        for _ in range(iteration):
            new_active_nodes = set()
            for node in active_nodes:
                new_active_nodes.update(G.neighbors(node))
            active_nodes.update(new_active_nodes)

        node_colors = [
            "red" if node in active_nodes else "lightgray" for node in G.nodes()
        ]
        ax = axes[iteration // cols, iteration % cols]
        ax.clear()
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            edge_color="gray",
            ax=ax,
        )
        ax.set_title(f"Iteration {iteration + 1}")

    for i in range(iteration + 1, rows * cols):
        fig.delaxes(axes[i // cols, i % cols])

    plt.tight_layout()
    plt.savefig(filename, dpi=720)
    plt.close()


def save_node_influence_in_seed_set(G, seed_set, filename):
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 8))

    influence_scores = {node: G.degree(node) for node in seed_set}

    nx.draw(G, pos, node_color="lightgray", node_size=500, edge_color="gray")
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=seed_set,
        node_color="red",
        node_size=[influence_scores[node] * 100 for node in seed_set],
    )
    nx.draw_networkx_labels(G, pos)

    plt.title("Node Influence in Seed Set")
    plt.savefig(os.path.join(output_dir, filename), dpi=720)
    plt.close()


def save_node_influence_seed_and_shortest_paths(G, seed_set, filename):
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 8))

    influence_scores = {node: G.degree(node) for node in G.nodes()}

    path_nodes = set()
    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            try:
                path = nx.shortest_path(G, source=seed_set[i], target=seed_set[j])
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue

    nx.draw(G, pos, node_color="lightgray", node_size=500, edge_color="gray")
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=seed_set,
        node_color="red",
        node_size=[influence_scores[node] * 100 for node in seed_set],
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=path_nodes,
        node_color="blue",
        node_size=[influence_scores[node] * 100 for node in path_nodes],
    )
    nx.draw_networkx_labels(G, pos)

    plt.title("Node Influence in Seed Set and Shortest Paths")
    plt.savefig(os.path.join(output_dir, filename), dpi=720)
    plt.close()


def calculate_propagation_probability(graph, node, seed_set, decay_factor=0.5):
    direct_neighbors = set(graph.neighbors(node))
    direct_influence = np.isin(list(direct_neighbors), seed_set).astype(float)

    second_order_neighbors = set()
    for neighbor in direct_neighbors:
        second_order_neighbors.update(graph.neighbors(neighbor))

    # Remove direct neighbors and the node itself from second order neighbors
    second_order_neighbors.difference_update(direct_neighbors)
    second_order_neighbors.discard(node)

    second_order_influence = np.isin(list(second_order_neighbors), seed_set).astype(
        float
    )

    # Apply decay factor for second-order influence
    total_influence = np.sum(direct_influence) + decay_factor * np.sum(
        second_order_influence
    )
    return total_influence


def calculate_worthiness(graph, node, seed_set, decay_factor=0.5):
    propagation_probability = calculate_propagation_probability(
        graph, node, seed_set, decay_factor
    )
    degree = len(list(graph.neighbors(node)))
    worthiness = propagation_probability * degree
    return worthiness


def evaluate_fitness(graph, seed_set, decay_factor=0.5):
    s_prime = set(seed_set)
    worthiness = np.array(
        [calculate_worthiness(graph, node, seed_set, decay_factor) for node in s_prime]
    )
    total_worthiness = np.sum(worthiness)
    if total_worthiness == 0:
        return 0.0
    proportions = worthiness / total_worthiness
    entropy = -np.sum(proportions * np.log(proportions + 1e-10))
    return entropy


def calculate_influence(graph, seed_node, decay_factor=0.5):
    influence_scores = {}
    for node in graph.nodes():
        influence_scores[node] = calculate_propagation_probability(
            graph, node, [seed_node], decay_factor
        )
    return influence_scores


def save_individual_node_influence(G, seed_set, decay_factor=0.5):
    pos = nx.kamada_kawai_layout(G)

    for node in seed_set:
        influence_scores = calculate_influence(G, node, decay_factor)
        max_influence = max(influence_scores.values(), default=1)
        norm_influence_scores = {
            n: influence_scores[n] / max_influence if max_influence else 0
            for n in G.nodes()
        }

        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))

        node_colors = [plt.cm.viridis(norm_influence_scores[n]) for n in G.nodes()]

        nx.draw(G, pos, node_color=node_colors, node_size=500, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="red", node_size=700)
        nx.draw_networkx_labels(G, pos)

        plt.title(f"Influence of Node {node}")
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_influence)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Influence Score")

        plt.savefig(
            os.path.join(output_dir, f"individual_node_influence_{node}.png"), dpi=720
        )
        plt.close()


def save_top_influential_shortest_path_nodes(G, seed_set, top_k, filename):
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 8))

    influence_scores = {node: G.degree(node) for node in G.nodes()}

    path_nodes = set()
    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            try:
                path = nx.shortest_path(G, source=seed_set[i], target=seed_set[j])
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue

    path_nodes -= set(seed_set)
    top_path_nodes = sorted(
        path_nodes, key=lambda node: influence_scores[node], reverse=True
    )[:top_k]

    nx.draw(G, pos, node_color="lightgray", node_size=500, edge_color="gray")
    nx.draw_networkx_nodes(
        G, pos, nodelist=seed_set, node_color="red", node_size=700, label="Seed Nodes"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=top_path_nodes,
        node_color="green",
        node_size=700,
        label="Top Path Nodes",
    )
    nx.draw_networkx_labels(G, pos)

    plt.title("Top Influential Nodes in Shortest Paths Between Seed Nodes")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename), dpi=720)
    plt.close()


def find_top_influential_nodes_in_shortest_paths(
    G, seed_set, top_k=3, decay_factor=0.5
):
    path_nodes = set()
    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            try:
                path = nx.shortest_path(G, source=seed_set[i], target=seed_set[j])
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue

    influence_scores = {
        node: calculate_propagation_probability(G, node, seed_set, decay_factor)
        for node in path_nodes
    }
    top_influential_nodes = sorted(
        influence_scores, key=influence_scores.get, reverse=True
    )[:top_k]

    return top_influential_nodes


def visualize_top_influential_nodes(G, seed_set, top_k=3, decay_factor=0.5):
    top_influential_nodes = find_top_influential_nodes_in_shortest_paths(
        G, seed_set, top_k, decay_factor
    )
    pos = nx.kamada_kawai_layout(G)

    for node in top_influential_nodes:
        influence_scores = calculate_influence(G, node, decay_factor)
        max_influence = max(influence_scores.values(), default=1)
        norm_influence_scores = {
            n: influence_scores[n] / max_influence for n in G.nodes()
        }

        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))

        node_colors = [plt.cm.viridis(norm_influence_scores[n]) for n in G.nodes()]

        nx.draw(G, pos, node_color=node_colors, node_size=500, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="red", node_size=700)
        nx.draw_networkx_labels(G, pos)

        plt.title(f"Influence of Node {node}")
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_influence)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Influence Score")

        plt.savefig(os.path.join(output_dir, f"iinfluence_node_{node}.png"), dpi=720)
        plt.close()


save_highlight_seed_neighbors(G, seed_set, "highlight_seed_neighbors.png")
save_highlight_shortest_paths(G, seed_set, "highlight_shortest_paths.png")
save_heatmap_node_influence(G, seed_set, "heatmap_node_influence.png")
save_replacement_process_visualization(G, seed_set, "replacement_process_step")
save_influence_convergence_visualization(
    G, seed_set, iterations=4, filename="influence_convergence.png"
)
save_node_influence_in_seed_set(G, seed_set, "node_influence_in_seed_set.png")
save_node_influence_seed_and_shortest_paths(
    G, seed_set, "node_influence_seed_and_shortest_paths.png"
)
save_individual_node_influence(G, seed_set)
save_top_influential_shortest_path_nodes(
    G, seed_set, top_k=3, filename="top_influential_shortest_path_nodes.png"
)
visualize_top_influential_nodes(G, seed_set, top_k=3)
