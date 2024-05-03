import networkx as nx


class Node:
    """
    Represents a node (user) in the social network.

    Attributes:
        id (int): Unique identifier for the node.
        initial_influence (float, optional): Initial influence probability
            for the Independent Cascade diffusion model. Defaults to None.
    """

    def __init__(self, node_id: int, initial_influence: float = None) -> None:
        """
        Initializes a Node object.

        Args:
            node_id (int): Unique identifier for the node.
            initial_influence (float, optional): Initial influence probability
                for the Independent Cascade diffusion model. Defaults to None.
        """

        self.id = node_id
        self.initial_influence = initial_influence

    def __str__(self) -> str:
        """
        Returns a string representation of the node.

        Returns:
            str: A string representation of the node in the format "Node {id}".
        """

        return f"Node {self.id}"

    def __repr__(self) -> str:
        """
        Returns a string representation of the node for debugging purposes.

        Returns:
            str: A string representation of the node in the format "Node {id}".
        """

        return f"Node {self.id}"


class Network:
    """
    Represents a social network as a graph using NetworkX.

    Attributes:
        directed (bool): Flag indicating whether the network is directed or undirected.
        graph (networkx.Graph or networkx.DiGraph): The underlying NetworkX graph object.
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initializes a Network object.

        Args:
            directed (bool, optional): Flag indicating whether the network is directed
                or undirected. Defaults to False (undirected).
        """

        self.directed = directed
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

    def add_node(self, node: Node) -> None:
        """
        Adds a node to the network.

        Args:
            node (Node): The node object to be added to the network.
        """

        self.graph.add_node(node.id, node=node)  # Store the entire Node object

    def add_edge(self, source: Node, target: Node) -> None:
        """
        Adds an edge between two nodes in the network.

        Args:
            source (Node): The source node of the edge.
            target (Node): The target node of the edge.
        """

        self.graph.add_edge(source.id, target.id)

    def get_nodes(self) -> list[Node]:
        """
        Returns a list of all nodes in the network.

        Returns:
            list[Node]: A list containing all Node objects in the network.
        """

        return [data["node"] for _, data in self.graph.nodes(data=True)]

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
