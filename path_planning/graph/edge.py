class Edge:
    def __init__(self, vertex1, vertex2):
        self._vertex1 = vertex1
        self._vertex2 = vertex2

    def get_vertices(self):
        return self._vertex1, self._vertex2

    def get_other_vertex(self, vertex):
        if vertex == self._vertex1:
            return self._vertex2
        elif vertex == self._vertex2:
            return self._vertex1
        else:
            return None
