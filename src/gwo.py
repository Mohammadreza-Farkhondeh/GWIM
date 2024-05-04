class SeedSet:
    """
    Represents a set of nodes chosen as the initial influencers in the network.

    Attributes:
        node_indices (list[int]): List of indices of the nodes in the seed set.
    """

    def __init__(self, node_indices: list[int]) -> None:
        """
        Initializes a SeedSet object.

        Args:
            node_indices (list[int]): List of indices of the nodes in the seed set.
        """

        self.node_indices = node_indices

    def get_seed_set(self) -> list[int]:
        """
        Returns the list of node indices in the seed set.

        Returns:
            list[int]: A list containing the indices of the nodes in the seed set.
        """

        return self.node_indices

    def set_seed_set(self, node_indices: list[int]) -> None:
        """
        set the list of node indices of the seed set.

        Args:
            node_indices list[int]: A list containing the indices of the nodes to be in the seed set.
        """

        self.node_indices = node_indices
