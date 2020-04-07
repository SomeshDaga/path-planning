import numpy as np

from robot import Robot
from utils import angles


class VectorController:
    def __init__(self, robot, k_w=5.0, max_linear_vel=1.0, max_angular_vel=1.0):
        self._robot = robot
        self._k_w = k_w
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel

    # Assumes differential drive controllers
    def loop(self, motion_vec=np.array([0, 0]), baseline=0.5, dt=0.1):
        # Get the motion model type
        motion_model = (self._robot.get_motion_model()).__name__

        if motion_model not in ['diff_drive_1', 'diff_drive_2']:
            raise("Invalid motion model '{0}' for VectorController".format(motion_model))

        # Get current pose of robot
        pose = self._robot.get_pose()

        # Get linear velocity by projecting the desired motion vector onto the current heading
        v = motion_vec[0] * np.cos(pose[2]) + motion_vec[1] * np.sin(pose[2])

        # Get the angle error between the desired heading and current heading
        angle_error = angles.wrap_angle_diff(np.arctan2(motion_vec[1], motion_vec[0]) - pose[2])
        w = self._k_w * angle_error

        # Enforce velocity limits
        if abs(v) > self._max_linear_vel:
            v = self._max_linear_vel * np.sign(v)
        if abs(w) > self._max_angular_vel:
            w = self._max_angular_vel * np.sign(w)

        if motion_model == 'diff_drive_1':
            self._robot.propagate(np.array([v, w]), dt=dt)
        elif motion_model == 'diff_drive_2':
            v_l = v - (w * baseline) / 2
            v_r = v + (w * baseline) / 2
            self._robot.propagate(np.array([v_l, v_r]), dt=dt, baseline=baseline)

