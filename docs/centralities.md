# Some common **Centrality Measures**

## Their significance:

- Degree Centrality: It is the simplest measure and is based on the number of connections a node has. Nodes with a high degree centrality are highly connected within the network. Degree centrality is useful for **identifying popular nodes** in social networks or highly connected entities in a transportation network.

- Closeness Centrality: It measures how close a node is to all other nodes in the network. Nodes with high closeness centrality can quickly interact with other nodes in the network. This measure is valuable in transportation or telecommunication networks to **identify nodes that can efficiently disseminate information or goods**.

- Betweenness Centrality: It quantifies the extent to which a node lies on the shortest paths between other nodes in the network. **Nodes with high betweenness centrality act as bridges or intermediaries facilitating communication between other nodes**. This measure is crucial in understanding the flow of information, traffic, or resources in a network, such as in transportation or social networks.

- Eigenvector Centrality: It measures the influence of a node in the network based on the connections of its neighbors. Nodes with high eigenvector centrality are connected to other influential nodes, thus indicating their importance in the network. This measure is often used in social network analysis to **identify individuals who have connections to other influential individuals**.

- PageRank: Initially developed by Google to rank web pages, PageRank assigns a score to each node based on the importance of the nodes linking to it. Nodes with high PageRank scores are considered **important or authoritative within the network**. PageRank is useful in web search algorithms, citation networks, and recommendation systems.

- Katz Centrality: It is a generalization of eigenvector centrality that considers the contributions of nodes at different distances. Katz centrality assigns higher scores to nodes with **direct connections and also to nodes that are reachable through paths of varying lengths**. This measure is applicable in various contexts, including social networks and citation networks.

These centrality measures provide valuable insights into the structure and functioning of networks, helping to identify key nodes, understand information flow, and predict network behavior. Depending on the specific characteristics of the network and the research questions, different centrality measures may be more appropriate to use.

## Their performance
- Degree Centrality: This measure tends to be one of the fastest because it simply involves counting the number of edges incident to each node.

- Closeness Centrality: Calculating closeness centrality involves finding shortest paths from each node to all other nodes in the network, which can be computationally intensive, especially for large networks with many nodes.

- Betweenness Centrality: This measure involves computing shortest paths between all pairs of nodes in the network and then counting the number of times each node lies on these paths. It can be computationally expensive, especially for large networks with dense connectivity.

- Eigenvector Centrality: Computing eigenvector centrality involves finding the dominant eigenvector of the adjacency matrix of the network, which can be computationally expensive, particularly for large networks.

- PageRank: PageRank involves iteratively computing the importance score of each node based on the importance scores of its neighbors. It can be relatively computationally expensive, especially for large networks with many nodes and edges.

### Benchmark results
1. 1K edges:

        degree_centrality: 0.0013742446899414062 seconds
        closeness_centrality: 1.1773898601531982 seconds
        betweenness_centrality: 37.61142563819885 seconds
        eigenvector_centrality: 0.0763700008392334 seconds
        pagerank: 0.23177123069763184 seconds

2. 10K edges:

        degree_centrality: 0.012479305267333984 seconds
        closeness_centrality: 175.5543246269226 seconds
        betweenness_centrality: Failed
        eigenvector_centrality: 6.574203729629517 seconds
        pagerank: 17.30572748184204 seconds
