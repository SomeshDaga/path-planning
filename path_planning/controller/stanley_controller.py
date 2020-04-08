import numpy as np

from robot import Robot
from utils import angles


class StanleyController:
    def __init__(self, robot,
                 linear_vel=1.0,
                 max_steering_angle=np.pi/4,
                 lateral_goal_tol=1.0,
                 k=1.0):
        self._robot = robot
        # Linear velocity is constant for a Pure Pursuit Controller
        self._linear_vel = linear_vel
        self._max_steering_angle = max_steering_angle
        self._lateral_goal_tol = lateral_goal_tol
        self._k = k

        self._x = None
        self._y = None
        self._goal_theta = None
        self._path_idx = 0
        self._has_path = False

    def set_path(self, x, y, goal_theta):
        # Ensure proper dimensions for path
        assert len(x) == len(y)

        self._x = x
        self._y = y
        self._path_idx = 0
        self._goal_theta = goal_theta
        self._has_path = True

    def finished(self):
        if not self._has_path:
            return True

        # The goal position are the last positions in the path
        goal_pos = np.array([self._x[-1], self._y[-1]])

        # Calculate lateral position to goal
        pose = self._robot.get_pose()
        last_seg_dir = np.array([self._x[-1] - self._x[-2], self._y[-1] - self._y[-2]])
        last_seg_dir = last_seg_dir / np.linalg.norm(last_seg_dir)
        ortho_dir = np.array([-last_seg_dir[1], last_seg_dir[0]])
        long_dist = (goal_pos - pose[0:2]).dot(last_seg_dir)
        lat_dist = (goal_pos - pose[0:2]).dot(ortho_dir)
        # If robot is at the end of the path, then we are done
        if long_dist <= 0 and abs(lat_dist) < self._lateral_goal_tol:
            return True
        else:
            return False

    def loop(self, dt=0.1, **kwargs):
        if not self.finished():
            motion_model = (self._robot.get_motion_model()).__name__
            if motion_model not in ['bicycle_1']:
                raise ("Invalid motion model '{0}' for Stanley Controller".format(motion_model))

            # Calculate lookahead point on reference path
            lookahead_pt, track_error, angle_error = self._get_lookahead_point(dt)

            if lookahead_pt is None:
                print("Hmmm...This is an unexpected error")
                return

            delta = angle_error + np.arctan2(self._k * track_error, self._linear_vel)
            if abs(delta) > self._max_steering_angle:
                delta = np.sign(delta) * self._max_steering_angle

            self._robot.propagate(np.array([self._linear_vel, delta]), dt=dt, **kwargs)

            return lookahead_pt

        return None

    def _get_lookahead_point(self, dt):
        # Get current robot x,y position
        pose = self._robot.get_pose()
        x, y = pose[0], pose[1]

        # Find the shortest distance from the current position of the robot
        # to the line segment at the current path segment and the next path segment
        lowest_error = np.inf
        angle_error = 0
        best_lookahead = None
        for path_idx in range(self._path_idx, self._path_idx + 2):
            if path_idx > len(self._x) - 2:
                break
            xs, ys = self._x[path_idx], self._y[path_idx]
            xe, ye = self._x[path_idx + 1], self._y[path_idx + 1]
            w1 = np.array([xe - xs, ye - ys])
            w1 = w1 / np.linalg.norm(w1)
            w2 = np.array([-w1[1], w1[0]])
            w2 = w2 / np.linalg.norm(w2)
            # lookahead_pt = np.array([xs, ys]) + (-np.array([xs, ys]) + pose[0:2]).dot(w1) * w1
            lookahead_pt = pose[0:2] + (np.array([xe, ye]) - pose[0:2]).dot(w2) * w2
            dist_to_lookahead = (lookahead_pt - pose[0:2]).dot(w2)

            # Check if we have driven past the line segment
            # Because of time discretization, we might fly off path segments. Adjust tolerances appropriately
            tol = self._linear_vel * dt
            x_min, x_max = min(xs, xe), max(xs, xe)
            y_min, y_max = min(ys, ye), max(ys, ye)
            on_path_segment = (x_min - tol) <= lookahead_pt[0] <= (x_max + tol) and \
                              (y_min - tol) <= lookahead_pt[1] <= (y_max + tol)

            if on_path_segment and dist_to_lookahead < lowest_error:
                lowest_error = dist_to_lookahead
                angle_error = angles.wrap_angle_diff(np.arctan2(w1[1], w1[0]) - pose[2])
                best_lookahead = lookahead_pt
                self._path_idx = path_idx

        return best_lookahead, lowest_error, angle_error
