import numpy as np

from utils import angles


def bicycle_1(x, u, axle_len=2.0, dt=0.1):
    # u[0] = linear velocity
    # u[1] = steering angle
    x = x + np.array([dt * u[0] * np.cos(x[2]),
                      dt * u[0] * np.sin(x[2]),
                      dt * (u[0] / axle_len) * np.tan(u[1])])
    x[2] = angles.wrap_angle(x[2], -np.pi, inclusive=False)

    return x


def bicycle_2(x, u, axle_len=2.0, dt=0.1):
    # u[0] = linear velocity
    # u[1] = steering angular velocity
    x = x + np.array([dt * u[0] * np.cos(x[2]),
                      dt * u[0] * np.sin(x[2]),
                      dt * (u[0] / axle_len) * np.tan(x[3]),
                      dt * u[1]])
    x[2] = angles.wrap_angle(x[2], -np.pi, inclusive=False)
    x[3] = angles.wrap_angle(x[3], -np.pi, inclusive=False)

    return x


def diff_drive_1(x, u, dt=0.1):
    # u[0] = linear velocity
    # u[1] = angular velocity
    x = x + np.array([dt * u[0] * np.cos(x[2] + dt * u[1] / 2),
                      dt * u[0] * np.sin(x[2] + dt * u[1] / 2),
                      dt * u[1]])
    x[2] = angles.wrap_angle(x[2], -np.pi, inclusive=False)

    return x


def diff_drive_2(x, u, baseline=0.5, dt=0.1):
    # u[0] = left wheel velocity
    # u[1] = right wheel velocity
    v = (u[0] + u[1]) / 2
    w = (u[1] - u[0]) / baseline
    x = diff_drive_1(x, [v, w], dt=dt)

    return x
