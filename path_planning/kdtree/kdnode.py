class KDNode:
    def __init__(self, vertex, split_dim):
        self._vertex = vertex
        self._split_dim = split_dim
        self.left = None
        self.right = None

    def get_split_dim(self):
        return self._split_dim

    def get_vertex(self):
        return self._vertex

    def __eq__(self, other):
        if other is None:
            return False
        else:
            return self._vertex == other.get_vertex()