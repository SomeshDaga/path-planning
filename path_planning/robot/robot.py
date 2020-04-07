import numpy as np

from utils import angles


class Robot:
    def __init__(self, pose, motion_model):
        self._pose = np.array(pose)
        self._motion_model = motion_model

    def is_at(self, point, tol=0.1):
        if np.linalg.norm(self._pose[0:2] - point) <= tol:
            return True
        else:
            return False

    def is_facing(self, point, tol=0.1):
        x_diff, y_diff = point[0] - self._pose[0], point[1] - self._pose[1]
        heading = np.arctan2(y_diff, x_diff)
        return self.has_heading(heading, tol=tol)

    def has_heading(self, heading, tol=0.1):
        if abs(angles.wrap_angle_diff(self._pose[2] - heading)) <= tol:
            return True
        else:
            return False

    def propagate(self, u, dt=0.1, **kwargs):
        self._pose = self._motion_model(self._pose, u, dt=dt, **kwargs)

    def set_pose(self, pose):
        self._pose = np.array(pose)

    def get_pose(self):
        return self._pose

    def get_motion_model(self):
        return self._motion_model
