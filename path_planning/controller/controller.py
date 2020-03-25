import numpy as np

from robot import Robot


class Controller:
    def __init__(self, robot):
        self._robot = robot
        self._x = None
        self._y = None
        self._goal_theta = None
        self._has_path = False

    def _get_target_pos(self):
        return np.array([self._x[self._target_idx],
                         self._y[self._target_idx]])

    def _get_pos(self, x, y):
        return np.array([x, y])

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

            if not robot_at_goal:
                self._robot.set_desired_motion(target_pos - self._robot.get_pose()[0:2])
                rotate_only = not self._robot.is_facing(target_pos)
            else:
                self._robot.set_desired_motion(np.array([np.cos(self._goal_theta),
                                                         np.sin(self._goal_theta)]))
                rotate_only = True

            self._robot.propagate(dt=dt, rotation_only=rotate_only)

