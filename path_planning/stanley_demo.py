import configparser
import matplotlib.pyplot as plt
import numpy as np

from controller import StanleyController
from models.motion_models import diff_drive_1, bicycle_1, bicycle_2
from robot import Robot


# Load configuration settings
config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == "__main__":
    # Load Configuration Settings
    # Simulation Settings
    dt = config.getfloat('SIM', 'dt')

    # Controller setting
    linear_vel = config.getfloat('STANLEY', 'linear_vel')
    max_steering_angle = config.getfloat('STANLEY', 'max_steering_angle') * np.pi / 180
    k = config.getfloat('STANLEY', 'k')
    lateral_goal_tol = config.getfloat('STANLEY', 'lateral_goal_tol')

    # Robot Settings
    axle_len = config.getfloat('ROBOT', 'axle_len')

    waypoints = np.array([[0, 0],
                          [8, 8],
                          [12, 4],
                          [9, 2]])


    # Initialize robot to starting position with zero heading
    # and initialize its controller
    # IMPORTANT (Use appropriate model)
    #   Bicycle model 1 is controlled by steering angle
    #   Bicycle model 2 is controlled by steering angular velocity
    motion_model = bicycle_1
    robot_pose = [waypoints[0][0], waypoints[0][1], 0.0]
    robot = Robot(robot_pose,
                  motion_model)
    controller = StanleyController(robot,
                                   linear_vel=linear_vel,
                                   max_steering_angle=max_steering_angle,
                                   lateral_goal_tol=lateral_goal_tol,
                                   k=k)
    controller.set_path(waypoints[:, 0], waypoints[:, 1], 0.0)

    pose_hist = np.empty((0, len(robot_pose)))
    plt.figure()
    while not controller.finished():
        plt.cla()
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'rx-', label="Waypoints")
        lookahead_pt = controller.loop(dt=0.1, axle_len=axle_len)
        pose = robot.get_pose()
        pose_hist = np.vstack((pose_hist, pose))
        if lookahead_pt is not None:
            plt.plot(lookahead_pt[0], lookahead_pt[1], 'bx', label='Lookahead Point')
        plt.plot(pose[0], pose[1], 'bo', markersize=11.25, label='Robot')
        plt.quiver(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]), color='red', label='Heading')
        plt.legend()
        plt.grid(True)
        plt.pause(0.001)

    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'rx-', label="Reference Path")
    plt.plot(pose_hist[:, 0], pose_hist[:, 1], 'k-.', label='Actual Trajectory')
    skip = 10
    plt.quiver(pose_hist[::skip, 0], pose_hist[::skip, 1], np.cos(pose_hist[::skip, 2]), np.sin(pose_hist[::skip, 2]),
               color='blue', units='inches', scale=4.0, zorder=3, label='Heading')
    plt.title('Stanley')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    plt.legend()
    plt.show()
