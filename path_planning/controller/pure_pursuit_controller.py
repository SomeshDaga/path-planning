import numpy as np

from robot import Robot
from utils import angles


class PurePursuitController:
    def __init__(self, robot,
                 linear_vel=0.5,
                 lookahead_dist=.25,
                 k_p=1.0,
                 k_d=0.0,
                 k_i=0.0):
        self._robot = robot
        self._lookahead_dist = lookahead_dist
        # Linear velocity is constant for a Pure Pursuit Controller
        self._linear_vel = linear_vel
        self._k_p = k_p
        self._k_d = k_d
        self._k_i = k_i

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

        # If robot is at the end of the path, then we are done
        if self._robot.is_at(goal_pos):
            return True
        else:
            return False

    def loop(self, dt=0.1):
        if not self.finished():
            motion_model = (self._robot.get_motion_model()).__name__

            if motion_model not in ['diff_drive_1', 'bicycle_1']:
                raise ("Invalid motion model '{0}' for VectorController".format(motion_model))

            # Calculate lookahead point on reference path
            lookahead_pt, track_error = self._get_lookahead_point()

            if lookahead_pt is None:
                print("Lookahead point not found...Try increasing lookahead distance")
                return

            pose = self._robot.get_pose()

            angle_error =\
                angles.wrap_angle_diff(np.arctan2(lookahead_pt[1] - pose[1], lookahead_pt[0] - pose[0]) - pose[2])
            w = self._k_p * angle_error
            if abs(w) > 5.0:
                w = np.sign(w) * 5.0
            self._robot.propagate(np.array([self._linear_vel, w]), dt=dt, axle_len=0.5)

            return lookahead_pt

        return None

    def _get_lookahead_point(self):
        # Get current robot x,y position
        pose = self._robot.get_pose()
        x, y = pose[0], pose[1]

        # Check if we are close to the end point of the path, if so, choose that
        dist_to_end = np.sqrt((x - self._x[-1]) ** 2 + (y - self._y[-1]) ** 2)
        if dist_to_end < self._lookahead_dist:
            self._path_idx = len(self._x) - 2
            lookahead = np.array([self._x[-1], self._y[-1]])
        else:
            lookahead = None
            # Iterate over all segments of the path
            for i in range(len(self._x) - 1):
                xs, ys = self._x[i], self._y[i]
                xe, ye = self._x[i + 1], self._y[i + 1]

                # Translate based on current position
                xs, ys = xs - x, ys - y
                xe, ye = xe - x, ye - y
                r = self._lookahead_dist

                # Check if circle of radius r intersects the segment
                dx, dy = xe - xs, ye - ys
                if dx != 0:
                    d = np.sqrt(dx ** 2 + dy ** 2)
                    D = xs * ye - xe * ys
                    discriminant = ((r*d) ** 2) - (D ** 2)

                    if discriminant < 0:
                        continue

                    # Get intersection points
                    x1, x2 = ((dy * D) - dx * np.sqrt(discriminant)) / (d ** 2), \
                             ((dy * D) + dx * np.sqrt(discriminant)) / (d ** 2)
                    y1, y2 = (dy / dx) * x1 - (D / dx), (dy / dx) * x2 - (D / dx)
                else:
                    k = xs
                    if abs(k) > r:
                        continue

                    # Get intersection points
                    x1 = x2 = xs
                    y1 = np.sqrt(r * r - k * k)
                    y2 = -y1

                # Check if either/both intersection points lie within the given segment
                x_min, x_max = min(xs, xe), max(xs, xe)
                y_min, y_max = min(ys, ye), max(ys, ye)
                print([x1 + x, y1 + y], [x2 + x, y2 + y], i)
                print([x_min + x, x_max + x], [y_min + y, y_max + y])
                tol = 1e-5
                valid_pts = list([a, b] for a, b in zip([x1, x2], [y1, y2])
                                 if (x_min - tol <= a <= x_max + tol and y_min - tol <= b <= y_max + tol))
                print(valid_pts)

                # Choose the point that is closer to the end of the segment
                best_dist = np.inf
                for pt in valid_pts:
                    # print("Valid pts: {0}, Segment: {1}".format(valid_pts, i))
                    dist = np.sqrt((pt[0] - xe) ** 2 + (pt[1] - ye) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        self._path_idx = i
                        lookahead = np.array([pt[0] + x, pt[1] + y])

        cross_track_error = None
        if lookahead is not None:
            # Calculate the cross track error
            # Get the start and end positions of the segment that the lookahead point is on
            xs, ys = self._x[self._path_idx], self._y[self._path_idx]
            xe, ye = self._x[self._path_idx + 1], self._y[self._path_idx + 1]
            dx, dy = xe - xs, ye - ys

            # Rotate w1 CCW by 90 deg
            w2 = np.array([-dy, dx])
            w2 = w2 / np.linalg.norm(w2)
            cross_track_error = np.array(lookahead - np.array([x, y])).dot(w2)

        return lookahead, cross_track_error
