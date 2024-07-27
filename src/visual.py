import os

import matplotlib.pyplot as plt
import networkx as nx

# Create a sample graph
G = nx.erdos_renyi_graph(30, 0.1, seed=42)
seed_set = [0, 1, 2]

# Output directory
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)


def save_highlight_seed_neighbors(G, seed_set, filename):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

    neighbors = set()
    for seed in seed_set:
        neighbors.update(G.neighbors(seed))
    neighbors -= set(seed_set)

    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=seed_set, node_color="red", node_size=700)
    nx.draw_networkx_nodes(
        G, pos, nodelist=neighbors, node_color="orange", node_size=500
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title("Seed Set and Their Neighbors")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_highlight_shortest_paths(G, seed_set, filename):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

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
    nx.draw_networkx_nodes(G, pos, nodelist=seed_set, node_color="red", node_size=700)
    nx.draw_networkx_nodes(
        G, pos, nodelist=path_nodes, node_color="blue", node_size=500
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title("Nodes in Shortest Paths Between Seed Nodes")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_heatmap_node_influence(G, seed_set, filename):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(figsize=(12, 10))

    # Compute influence as the degree for visualization
    influence_scores = {node: G.degree(node) for node in G.nodes()}
    max_influence = max(influence_scores.values(), default=1)
    norm_influence_scores = {
        node: (influence_scores[node] / max_influence) for node in G.nodes()
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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_replacement_process_visualization(G, seed_set, filename):
    pos = nx.spring_layout(G, seed=42)

    for i, node in enumerate(seed_set):
        plt.figure(figsize=(12, 10))

        neighbors = list(G.neighbors(node))
        neighbor_degrees = {neighbor: G.degree(neighbor) for neighbor in neighbors}

        nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500)
        nx.draw_networkx_nodes(
            G, pos, nodelist=seed_set, node_color="red", node_size=700
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=neighbors, node_color="orange", node_size=500
        )

        if neighbor_degrees:
            highest_degree_neighbor = max(neighbor_degrees, key=neighbor_degrees.get)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[highest_degree_neighbor],
                node_color="blue",
                node_size=900,
            )

        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos)

        plt.title(f"Replacement Process Step {i + 1}")
        plt.savefig(os.path.join(output_dir, f"{filename}_{i + 1}.png"))
        plt.close()


def save_influence_convergence_visualization(
    G, seed_set, iterations=10, filename="convergence.png"
):
    pos = nx.spring_layout(G, seed=42)

    for iteration in range(iterations):
        plt.figure(figsize=(12, 10))

        active_nodes = set(seed_set)
        for _ in range(iteration):
            new_active_nodes = set()
            for node in active_nodes:
                new_active_nodes.update(G.neighbors(node))
            active_nodes.update(new_active_nodes)

        node_colors = [
            "red" if node in active_nodes else "lightgray" for node in G.nodes()
        ]
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            edge_color="gray",
        )

        plt.title(f"Influence Spread at Iteration {iteration + 1}")
        plt.savefig(os.path.join(output_dir, f"{filename}_{iteration + 1}.png"))
        plt.close()


def save_node_influence_in_seed_set(G, seed_set, filename):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_node_influence_seed_and_shortest_paths(G, seed_set, filename):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

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

    plt.title("Node Influence in Seed Set and Shortest Path Nodes")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_individual_node_influence(G, seed_set):
    pos = nx.spring_layout(G, seed=42)
    influence_scores = {node: G.degree(node) for node in G.nodes()}

    for node in seed_set:
        plt.figure(figsize=(12, 10))

        nx.draw(G, pos, node_color="lightgray", node_size=500, edge_color="gray")
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color="red",
            node_size=influence_scores[node] * 100,
        )
        nx.draw_networkx_labels(G, pos)

        plt.title(f"Influence of Node {node}")
        plt.savefig(os.path.join(output_dir, f"influence_node_{node}.png"))
        plt.close()


def save_top_influential_shortest_path_nodes(G, seed_set, top_k=3):
    pos = nx.spring_layout(G, seed=42)
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
    top_influential_nodes = sorted(
        path_nodes, key=lambda node: influence_scores[node], reverse=True
    )[:top_k]

    for i, node in enumerate(top_influential_nodes):
        plt.figure(figsize=(12, 10))

        nx.draw(G, pos, node_color="lightgray", node_size=500, edge_color="gray")
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color="blue",
            node_size=influence_scores[node] * 100,
        )
        nx.draw_networkx_labels(G, pos)

        plt.title(f"Top Influential Node in Shortest Paths: {node}")
        plt.savefig(os.path.join(output_dir, f"top_influential_node_{i + 1}.png"))
        plt.close()


def calculate_influence(G, seed_set, node):
    extended_seed_set = set(seed_set) | {node}

    node_degrees = dict(G.degree())

    influence = 0
    for n in G.nodes():
        if n in extended_seed_set:
            neighbors = set(G.neighbors(n))
            influence += sum(
                node_degrees.get(nei, 0)
                for nei in neighbors
                if nei in extended_seed_set
            )

    return influence


def visualize_influence(G, seed_set, imname):
    pos = nx.spring_layout(G, seed=42)
    influences = {node: calculate_influence(G, seed_set, node) for node in seed_set}
    max_influence = max(influences.values(), default=1)

    plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(figsize=(12, 10))
    node_colors = [influences.get(node, 0) / max_influence for node in G.nodes()]
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        edge_color="gray",
        cmap=plt.cm.viridis,
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_influence)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Influence Score")

    plt.title("Heatmap of Node Influence")
    plt.savefig(os.path.join(output_dir, imname))
    plt.close()

    all_paths = []
    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            try:
                path = nx.shortest_path(G, source=seed_set[i], target=seed_set[j])
                all_paths.extend(path)
            except nx.NetworkXNoPath:
                continue

    path_nodes = set(all_paths)
    path_influences = {
        node: calculate_influence(G, seed_set, node) for node in path_nodes
    }
    top_nodes = sorted(path_influences, key=path_influences.get, reverse=True)[:3]

    plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(figsize=(12, 10))
    node_colors = [path_influences.get(node, 0) / max_influence for node in G.nodes()]
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        edge_color="gray",
        cmap=plt.cm.viridis,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=top_nodes,
        node_color="none",
        node_size=700,
        edgecolors="blue",
        linewidths=2,
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_influence)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Influence Score")

    plt.title("Top Influential Nodes in Shortest Paths")
    plt.savefig(os.path.join(output_dir, imname))
    plt.close()


# Save visualizations
save_highlight_seed_neighbors(G, seed_set, "highlight_seed_neighbors.png")
save_highlight_shortest_paths(G, seed_set, "highlight_shortest_paths.png")
save_heatmap_node_influence(G, seed_set, "heatmap_node_influence.png")
save_replacement_process_visualization(G, seed_set, "replacement_process")
save_influence_convergence_visualization(G, seed_set, iterations=10)
save_node_influence_in_seed_set(G, seed_set, "node_influence_in_seed_set.png")
save_node_influence_seed_and_shortest_paths(G, seed_set, "node_influence_seed.png")
save_individual_node_influence(G, seed_set)
save_top_influential_shortest_path_nodes(G, seed_set)
visualize_influence(G, seed_set, "influence_top_nodes.png")


def compute_centrality(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    return degree_centrality, betweenness_centrality, closeness_centrality


degree_centrality, betweenness_centrality, closeness_centrality = compute_centrality(G)


def get_best_nodes_in_shortest_paths(G, seed_set, top_k=3):
    path_nodes = set()
    for i in range(len(seed_set)):
        for j in range(i + 1, len(seed_set)):
            try:
                path = nx.shortest_path(G, source=seed_set[i], target=seed_set[j])
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue

    path_nodes -= set(seed_set)
    node_centralities = {
        node: (
            degree_centrality.get(node, 0),
            betweenness_centrality.get(node, 0),
            closeness_centrality.get(node, 0),
        )
        for node in path_nodes
    }

    top_nodes = sorted(
        node_centralities,
        key=lambda node: (
            node_centralities[node][0]
            + node_centralities[node][1]
            + node_centralities[node][2]
        ),
        reverse=True,
    )[:top_k]

    return top_nodes


top_k = len(seed_set)
best_nodes = get_best_nodes_in_shortest_paths(G, seed_set, top_k=top_k)


def prepare_plot_data(nodes, centrality_measures):
    node_labels = {node: f"Node {node}" for node in nodes}
    centralities = [centrality_measures[node] for node in nodes]
    return node_labels, centralities


seed_set_labels, seed_set_centralities = prepare_plot_data(seed_set, degree_centrality)
best_nodes_labels, best_nodes_centralities = prepare_plot_data(
    best_nodes, degree_centrality
)

fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
fig.suptitle("Centrality Measures Comparison", fontsize=16)

for idx, (centrality_name, centrality_values) in enumerate(
    [
        ("Degree", degree_centrality),
        ("Betweenness", betweenness_centrality),
        ("Closeness", closeness_centrality),
    ]
):
    ax = axs[0, idx]
    values = [centrality_values[node] for node in seed_set]
    labels = [f"Node {node}" for node in seed_set]
    ax.bar(labels, values, color="skyblue")
    ax.set_title(f"Seed Set - {centrality_name}")
    ax.set_xticklabels(labels, rotation=45, ha="right")

for idx, (centrality_name, centrality_values) in enumerate(
    [
        ("Degree", degree_centrality),
        ("Betweenness", betweenness_centrality),
        ("Closeness", closeness_centrality),
    ]
):
    ax = axs[1, idx]
    values = [centrality_values[node] for node in best_nodes]
    labels = [f"Node {node}" for node in best_nodes]
    ax.bar(labels, values, color="salmon")
    ax.set_title(f"Best Nodes - {centrality_name}")
    ax.set_xticklabels(labels, rotation=45, ha="right")

plt.savefig(os.path.join(output_dir, "multi.png"))
plt.close()
