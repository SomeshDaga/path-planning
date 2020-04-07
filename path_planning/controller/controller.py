import numpy as np

from robot import Robot
from utils import angles


class Controller:
    def __init__(self, robot, k_w=5.0, max_linear_vel=1.0, max_angular_vel=1.0):
        self._robot = robot
        self._k_w = k_w
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel
        self._x = None
        self._y = None
        self._goal_theta = None
        self._target_idx = 0
        self._has_path = False

    def set_path(self, x, y, goal_theta=None):
        # Ensure proper dimensions for path
        assert len(x) == len(y)

        self._x = x
        self._y = y
        self._goal_theta = goal_theta
        self._target_idx = 0
        self._has_path = True

    def finished(self):
        if not self._has_path:
            return True

        # The goal position are the last positions in the path
        goal_pos = np.array([self._x[-1], self._y[-1]])

        # Check if angular conditions (if any) are satisfied
        angle_satisfied = self._goal_theta is None or self._robot.has_heading(self._goal_theta)

        # If robot is at the end of the path with appropriate heading, then we are done
        if self._robot.is_at(goal_pos) and angle_satisfied:
            return True
        else:
            return False

    def loop(self, dt=0.1):
        if not self.finished():
            target_pos = np.array([self._x[self._target_idx],
                                   self._y[self._target_idx]])

            robot_at_goal = False
            if self._robot.is_at(target_pos):
                if self._target_idx < len(self._x) - 1:
                    self._target_idx += 1
                    target_pos = np.array([self._x[self._target_idx],
                                           self._y[self._target_idx]])
                else:
                    robot_at_goal = True

            pose = self._robot.get_pose()
            if not robot_at_goal:
                pose_diff = target_pos - pose[0:2]
                if not self._robot.is_facing(target_pos):
                    v = 0
                else:
                    v = pose_diff[0] * np.cos(pose[2]) + pose_diff[1] * np.sin(pose[2])

                w = self._k_w * angles.wrap_angle_diff(np.arctan2(pose_diff[1], pose_diff[0]) - pose[2])
            else:
                v = 0
                w = self._k_w * angles.wrap_angle_diff(self._goal_theta - pose[2])

            # Enforce velocity limits
            if abs(v) > v:
                v = np.sign(v) * self._max_linear_vel
            if abs(w) > w:
                w = np.sign(w) * self._max_angular_vel

            self._robot.propagate(np.array([v, w]), dt=dt)
