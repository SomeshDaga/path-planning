from enum import Enum
import numpy as np

from .astar import AStar
from graph import Vertex
from kdtree import KDTree
from map import Map

class Extend(Enum):
    TRAPPED = 0
    ADVANCED = 1
    REACHED = 2

class RRT:
    def __init__(self, map, inflation_radius=1.0):
        self._map = \
            Map(map.get_inflated_obstacles_data(inflation_radius=inflation_radius),
                resolution=map.get_resolution(),
                d_theta=map.get_delta_theta())

    def extend(self, tree, q, step=0.5):
        q_near = (tree.get_knn(q, k=1)[0][1]).get_vertex()

        distance = np.sqrt(q.calc_dist_2(q_near))
        if distance <= step:
            q_new = q
        else:
            frac = step / distance
            x_new = (1 - frac) * q_near.get_data()[0] + frac * q.get_data()[0]
            y_new = (1 - frac) * q_near.get_data()[1] + frac * q.get_data()[1]
            q_new = Vertex(np.array([x_new, y_new]))

        if self._map.can_connect(q_near.get_data(), q_new.get_data()):
            tree.add_vertex(q_new)
            q_near.add_neighbour(q_new)
            if q == q_new:
                return Extend.REACHED, q_new
            else:
                return Extend.ADVANCED, q_new

        return Extend.TRAPPED, None

    def connect(self, tree, q, step=0.5):
        while True:
            result, _ = self.extend(tree, q, step=step)
            if result != Extend.ADVANCED:
                return result

    def sample_point(self):
        # Get the width and height of the map
        # in world units
        width = self._map.get_width()
        height = self._map.get_height()

        # Generate samples within the (inflated) map
        sample = np.random.uniform(high=[width, height])
        return sample

    def plan(self, start, goal, iterations=1000, step=0.5):
        start = Vertex(start)
        goal = Vertex(goal)

        # Add the start and goal points to their appropriate trees
        tree_a = KDTree(dimension=2)
        tree_b = KDTree(dimension=2)
        tree_a.add_vertex(start)
        tree_b.add_vertex(goal)

        for i in range(iterations):
            q_rand = Vertex(self.sample_point())
            result, q_new = self.extend(tree_a, q_rand, step=step)

            if result != Extend.TRAPPED:
                if self.connect(tree_b, q_new, step=step) == Extend.REACHED:
                    heuristic = lambda vertex: np.sqrt(vertex.calc_dist_2(goal))
                    return AStar.find_path(start, goal, heuristic)

            tree_a, tree_b = tree_b, tree_a

