import numpy as np

class PotentialField:
    def __init__(self, goal, map, k_a=1.0, k_r=1.0, obs_dist_thresh=5.0):
        # Potential Field Parameters
        self._k_a = k_a
        self._k_r = k_r
        self._obs_dist_thresh = obs_dist_thresh

        # Reference to objects for planning
        self._goal = goal
        self._map = map

    def calc_attractive_force(self, pos):
        return self._k_a * (self._goal - pos)

    def calc_repulsive_force(self, pos):
        # Get vectors and distances to obstacles within
        # the obstacle threshold distance
        [obstacles, distances] = self._map.get_obstacles_in_radius(pos, self._obs_dist_thresh)
        repulsion_vec = np.array([0.0, 0.0])
        for obs, dist in zip(obstacles, distances):
            repulsion_vec += -1*self._k_r*(1/dist - 1/self._obs_dist_thresh)*(1/dist**3)*(obs - pos)

        return repulsion_vec

    def get_force(self, pos):
        return self.calc_attractive_force(pos) + \
               self.calc_repulsive_force(pos)