import matplotlib.pyplot as plt
import numpy as np

from controller import PurePursuitController
from models.motion_models import diff_drive_1, bicycle_1
from robot import Robot


if __name__ == "__main__":
    waypoints = np.array([[0, 0],
                          [4, 4],
                          [8, 4],
                          [9, 2]])


    # Initialize robot to starting position with zero heading
    # and initialize its controller
    motion_model = bicycle_1
    robot = Robot([waypoints[0][0], waypoints[0][1], 0.0],
                  motion_model)
    controller = PurePursuitController(robot)
    controller.set_path(waypoints[:, 0], waypoints[:, 1], 0.0)

    plt.figure()
    while not controller.finished():
        plt.cla()
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'rx-', label="Waypoints")
        lookahead_pt = controller.loop(dt=0.1)
        pose = robot.get_pose()
        if lookahead_pt is not None:
            plt.plot(lookahead_pt[0], lookahead_pt[1], 'bx', label='Lookahead Point')
        plt.plot(pose[0], pose[1], 'bo', markersize=11.25, label='Robot')
        plt.legend()
        plt.grid(True)
        plt.pause(0.001)

    plt.show()