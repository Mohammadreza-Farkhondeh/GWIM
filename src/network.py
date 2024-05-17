import networkx as nx


class Network:
    """
    Represents a social network as a graph using NetworkX.

    Attributes:
        directed (bool): Flag indicating whether the network is directed or undirected.
        graph (networkx.Graph or networkx.DiGraph): The underlying NetworkX graph object.
    """

    def __init__(
        self,
        graph: nx.Graph = None,
    ) -> None:
        """
        Initializes a Network object.

        Args:
            directed (bool, optional): Flag indicating whether the network is directed
                or undirected. Defaults to False (undirected).
        """
        if graph is not None:
            self.graph = graph

        self.v_prime: list[int] = self.get_v_prime()
        self.v_prime_size = len(self.v_prime)

    def add_node(self, node: int) -> None:
        """
        Adds a node to the network.

        Args:
            node (Node): The node object to be added to the network.
        """

        self.graph.add_node(node, node=node)

    def add_edge(self, source: int, target: int) -> None:
        """
        Adds an edge between two nodes in the network.

        Args:
            source (int): The source node id of the edge.
            target (int): The target node id of the edge.
        """

        self.graph.add_edge(source, target)

    def get_nodes(self) -> list[int]:
        """
        Returns a list of all nodes in the network.

        Returns:
            list[int]: A list containing all Node ids in the network.
        """
        return self.graph.nodes()

    def get_v_prime(self):
        """
        Returns the nodes with degree greater than 1 (V').

        Args:
            network (Network): The network object.

        Returns:
            int: The number of nodes with degree greater than 1 (V').
        """
        return [node for node, degree in self.graph.degree if degree > 1]

    def __str__(self) -> str:
        """
        Returns a string representation of the network summary.

        Returns:
            str: A string summarizing the network size (number of nodes and edges).
        """

        return f"Network: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"

    def __repr__(self) -> str:
        """
        Returns a string representation of the network for debugging purposes.

        Returns:
            str: A string representation similar to the __str__ method.
        """

        return f"Network: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"
