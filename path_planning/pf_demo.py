import configparser
import matplotlib.pyplot as plt
import numpy as np

from map import Map
from planners.pf import PotentialField
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

    # Potential Field Settings
    k_a = config.getfloat('PF', 'k_a')
    k_r = config.getfloat('PF', 'k_r')

    # Initialize map
    data = load_map(map_file)
    map = Map(data, resolution=map_resolution, d_theta=2*np.pi/num_scans)

    # Set start and goal positions
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])

    # Initialize algorithm
    pf = PotentialField(goal, map, k_a=k_a, k_r=k_r)

    # Initialize robot to starting position with zero heading
    robot = Robot(max_linear_vel=robot_max_linear_vel,
                  max_angular_vel=robot_max_angular_vel)
    robot.set_pose(start[0], start[1], 0.0)

    plt.figure()
    while True:
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        # Propagate robot using motion vector and get its pose
        robot.propagate(dt=dt)
        robot_pose = robot.get_pose()

        # Plot the live obstacles being detected on the map
        map_image = map.get_data()
        obstacles, distances = map.get_obstacles_in_radius(robot_pose[0:2], 5)
        for obs, d in zip(obstacles, distances):
            obs = map.worldToMap(obs)
            map_image[obs[1], obs[0]] = 128
        plt.imshow(map_image, extent=[0, 10, 10, 0])

        # Plot robot position
        plt.plot(robot_pose[0], robot_pose[1], 'rx', markersize=15)

        # Plot directions of attractive and repulsive forces
        r = pf.calc_repulsive_force(robot_pose[0:2])
        a = pf.calc_attractive_force(robot_pose[0:2])
        plt.quiver(robot_pose[0], robot_pose[1], a[0], -a[1], color='g')
        plt.quiver(robot_pose[0], robot_pose[1], r[0], -r[1], color='r')

        # Pause animation between frames
        plt.grid(True)
        plt.pause(0.01)

        # Get the motion vectors to apply to the robot for propagation
        # in the next update cycle
        motion_vec = pf.get_force(robot_pose[0:2])
        robot.set_desired_motion(motion_vec)
