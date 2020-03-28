import heapq
import numpy as np

from .kdnode import KDNode


class KDTree:
    def __init__(self, dimension=2):
        self._dimension = dimension
        self._root = None

    def add_vertex(self, vertex):
        self._root = self._add_vertex(self._root, vertex)

    def get_all_vertices(self):
        vertices = []
        self._get_vertices(self._root, vertices)
        return vertices

    def _get_vertices(self, root, vertices):
        if root is not None:
            self._get_vertices(root.left, vertices)
            vertices.append(root.get_vertex())
            self._get_vertices(root.right, vertices)

    def _add_vertex(self, root, vertex, split_dim=0):
        if root is None:
            root = KDNode(vertex, split_dim)
        elif root.get_vertex() == vertex:
            # Do nothing in this case, since the vertex
            # already exists
            pass
        else:
            if vertex.get_data()[split_dim] < \
               root.get_vertex().get_data()[split_dim]:
                root.left = self._add_vertex(root.left,
                                             vertex,
                                             self.next_split_dim(split_dim))
            else:
                root.right = self._add_vertex(root.right,
                                              vertex,
                                              self.next_split_dim(split_dim))
        return root

    def next_split_dim(self, current_dim):
        return np.mod(current_dim + 1, self._dimension)

    def get_knn(self, vertex, k=5):
        hpq = []
        self._find_knn(self._root, vertex, hpq, k)
        return hpq

    def _find_knn(self, root, vertex, hpq, k=5):
        if root is None:
            return

        # Traverse the right or left of the tree based on
        # the value of the vertex
        traversed_node = None
        if vertex.get_data()[root.get_split_dim()] < \
           root.get_vertex().get_data()[root.get_split_dim()]:
            self._find_knn(root.left,
                           vertex,
                           hpq,
                           k)
            traversed_node = root.left
        else:
            self._find_knn(root.right,
                           vertex,
                           hpq,
                           k)
            traversed_node = root.right

        if vertex != root.get_vertex():
            if len(hpq) < k:
                # Calculate inverse of distance squared to vertex
                inv_dist_2 = 1 / vertex.calc_dist_2(root.get_vertex())
                heapq.heappush(hpq, (inv_dist_2, root))
            else:
                # Push the node on the heap if it's distance is closer to the given vertex
                # than the worst node in the heap
                heapq.heappushpop(hpq, (1 / vertex.calc_dist_2(root.get_vertex()),
                                        root))

        if len(hpq) < k or abs(vertex.get_data()[root.get_split_dim()] -
                               root.get_vertex().get_data()[root.get_split_dim()]) < \
                           np.sqrt(1 / hpq[0][0]):
            if traversed_node == root.left:
                self._find_knn(root.right,
                               vertex,
                               hpq,
                               k)
            else:
                self._find_knn(root.left,
                               vertex,
                               hpq,
                               k)
