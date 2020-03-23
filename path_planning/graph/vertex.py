from .edge import Edge


class Vertex:
    def __init__(self, data):
        self._dimension = len(data)
        self._data = data
        self._neighbours = {}

    def get_data(self):
        return self._data

    def calc_dist_2(self, other):
        return (self._data[0] - other.get_data()[0]) ** 2 + \
               (self._data[1] - other.get_data()[1]) ** 2

    def get_dimension(self):
        return self._dimension

    def get_neighbours(self):
        return self._neighbours.keys()

    def add_neighbour(self, vertex):
        # Check if we already have the given vertex as a neighbour
        # and vice versa
        if not self.is_neighbour(vertex):
            self._neighbours[vertex] = Edge(self, vertex)
            # Add this vertex as a neighbour to the given vertex
            vertex.add_neighbour(self)

    def is_neighbour(self, vertex):
        return vertex in self._neighbours.keys()

    def __eq__(self, vertex):
        return (self._data == vertex.get_data()).all()

    def __hash__(self):
        return hash((self._data[0], self._data[1]))