import configparser
import matplotlib.pyplot as plt
import numpy as np

from controller import Controller
from map import Map
from planners import RRT
from robot import Robot
from utils.load_map import load_map

# Load configuration settings
config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == "__main__":
    # Load Configuration Settings
    # Simulation Settings
    dt = config.getfloat('SIM', 'dt')

    # Map Settings
    map_file = config.get('MAP', 'file')
    map_resolution = config.getfloat('MAP', 'resolution')
    num_scans = config.getint('MAP', 'num_scans')

    # Robot Settings
    robot_max_linear_vel = config.getfloat('ROBOT', 'max_linear_vel')
    robot_max_angular_vel = config.getfloat('ROBOT', 'max_angular_vel')

    # RRT Settings
    rrt_num_iterations = config.getint('RRT', 'num_iterations')
    rrt_step_length = config.getfloat('RRT', 'step_length')
    rrt_inflation_radius = config.getfloat('RRT', 'inflation_radius')

    # Initialize map
    data = load_map(map_file)
    map = Map(data, resolution=map_resolution, d_theta=2*np.pi/num_scans)

    # Set start position and waypoints
    start = np.array([7.0, 1.5])
    waypoints = np.array([[7.0, 1.5, 3*np.pi/2],
                          [9.0, 5.0, np.pi],
                          [3.0, 9.5, np.pi/2],
                          [0.5, 5.0, 0]])

    # Initialize algorithm
    rrt = RRT(map, inflation_radius=rrt_inflation_radius)

    # Initialize robot to starting position with zero heading
    # and initialize its controller
    robot = Robot(max_linear_vel=robot_max_linear_vel,
                  max_angular_vel=robot_max_angular_vel)
    robot.set_pose(start[0], start[1], 0.0)
    controller = Controller(robot)

    # Visit each waypoint sequentially
    plt.figure()
    for waypoint in waypoints:
        path = rrt.plan(robot.get_pose()[0:2],
                        waypoint[0:2],
                        iterations=rrt_num_iterations,
                        step=rrt_step_length)
        path_x = list(vertex.get_data()[0] for vertex in path)
        path_y = list(vertex.get_data()[1] for vertex in path)

        # Set the path for the controller
        controller.set_path(path_x, path_y, waypoint[2])

        while not controller.finished():
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.imshow(map.get_data(), extent=[0, 10, 10, 0])
            plt.plot(path_x, path_y, color='green', linestyle='dashed', linewidth=3)
            controller.loop(dt=dt)
            robot_pose = robot.get_pose()
            plt.plot(robot_pose[0], robot_pose[1], 'bx', markersize=10, markeredgewidth=2)
            plt.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[2]), -np.sin(robot_pose[2]),
                       color='r', scale=2.5, scale_units='inches')
            plt.grid(True)
            plt.pause(.0001)

    print("Reached Goal!")
