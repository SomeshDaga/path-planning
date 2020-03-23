import numpy as np

from utils import angles


class Robot:
    def __init__(self, k_w=5.0, max_linear_vel=1.0, max_angular_vel=1.0):
        self._pose = np.array([0.0, 0.0, 0.0])
        self._k_w = k_w
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel
        self._linear_vel = self._angular_vel = 0.0
        self._motion_vec = np.array([0.0, 0.0])

    def set_desired_motion(self, motion_vec):
        self._motion_vec = motion_vec

    def propagate(self, dt=0.1):
        angle = self._pose[2]

        # Set the control inputs
        # Linear velocity
        self._linear_vel = \
            self._motion_vec[0] * np.cos(angle) + \
            self._motion_vec[1] * np.sin(angle)
        if self._linear_vel < 0:
            self._linear_vel = max(self._linear_vel, -self._max_linear_vel)
        else:
            self._linear_vel = min(self._linear_vel, self._max_linear_vel)

        # Angular Velocity
        angle_error = \
            angles.wrap_angle_diff(np.arctan2(self._motion_vec[1], self._motion_vec[0]) - angle)
        self._angular_vel = self._k_w * angle_error
        if self._angular_vel < 0:
            self._angular_vel = max(self._angular_vel, -self._max_angular_vel)
        else:
            self._angular_vel = min(self._angular_vel, self._max_angular_vel)

        # Update pose based on motion model
        self._pose += np.array([dt * self._linear_vel * np.cos(angle + self._angular_vel * dt / 2),
                                dt * self._linear_vel * np.sin(angle + self._angular_vel * dt / 2),
                                dt * self._angular_vel])
        # Ensure angle is in the range (-pi, pi]
        self._pose[2] = angles.wrap_angle(self._pose[2], -np.pi, inclusive=False)

    def get_velocity(self):
        return self._linear_vel, self._angular_vel

    def set_pose(self, x, y, theta):
        self._pose = np.array([x, y, theta])

    def get_pose(self):
        return self._pose
