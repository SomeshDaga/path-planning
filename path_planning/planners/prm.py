import numpy as np

from graph import Vertex
from map import Map
from kdtree import KDTree

class PRM:
    def __init__(self, map, inflation_radius=1.0):
        # We want to use an inflated obstacle map
        # for PRM planning
        self._map = \
            Map(map.get_inflated_obstacles_data(inflation_radius=inflation_radius),
                resolution=map.get_resolution(),
                d_theta=map.get_delta_theta())
        self._kdtree = KDTree(dimension=2)
        self._vertices = []

    def generate_samples(self, num_samples=1000):
        # Get the width and height of the map
        # in world units
        width = self._map.get_width()
        height = self._map.get_height()

        samples = []
        sample_count = 0
        while sample_count < num_samples:
            sample = np.random.uniform(high=[width, height])
            if not self._map.is_occupied(sample):
                samples.append(sample)
                sample_count += 1
        samples = np.array(samples)

        # In order to create a balanced KD-Tree later, sort
        # the samples based on their distance from the center
        # of the map (in increasing order)
        center_w, center_h = width / 2, height / 2
        dist = list((sample[0] - center_w) ** 2 + (sample[1] - center_h) ** 2
                    for sample in samples)
        # Find indexes of the sample in increasing order of distance
        sort_idxs = np.argsort(dist)

        # Get the sorted samples
        return samples[sort_idxs]

    def generate_roadmap(self, num_samples=1000, num_neighbours=5):
        samples = self.generate_samples(num_samples=num_samples)
        for sample in samples:
            vertex = Vertex(sample)
            self._vertices.append(vertex)
            self._kdtree.add_vertex(vertex)

        for vertex in self._vertices:
            neighbours = self._kdtree.get_knn(vertex, k=num_neighbours)
            for n in neighbours:
                # Check if we can connect to the neighbour
                # without going through an obstacle
                if self._map.can_connect(vertex.get_data(),
                                         n[1].get_vertex().get_data()):
                    vertex.add_neighbour(n[1].get_vertex())

    def get_vertices(self):
        return self._vertices

    def find_shortest_path(self, start, goal):
        pass

