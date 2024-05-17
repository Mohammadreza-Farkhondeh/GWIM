import networkx as nx
import pandas as pd

from src.network import Network


def get_network(n: str) -> Network:
    datasets = {
        "soc-twitter-follows": "seed/soc-twitter-follows.csv",
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
    else:
        nodes = [1, 2, 3, 4, 5]
        edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
        network = Network(nodes=nodes, edges=edges)

    return network
