import matplotlib.pyplot as plt
import numpy as np

from map import Map
from planners.pf import PotentialField
from robot import Robot
from utils.load_map import load_map


if __name__ == "__main__":
    data, resolution = load_map()
    map = Map(data, resolution=resolution, d_theta=np.pi/24)

    # Set start and goal positions
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])

    # Initialize algorithm and robot
    pf = PotentialField(goal, map, k_a=15.0, k_r=10.0)
    robot = Robot()
    robot.set_pose(start[0], start[1], 0.0)

    plt.figure()

    while True:
        plt.cla()
        map_image = map.get_image()
        robot.propagate(dt=0.1)
        robot_pose = robot.get_pose()
        obstacles, distances = map.get_obstacles_in_radius(robot_pose[0:2], 5)
        for obs, d in zip(obstacles, distances):
            obs = map.worldToMap(obs)
            map_image[obs[1], obs[0]] = 128
        plt.imshow(map_image, extent=[0, 10, 10, 0])
        plt.plot(robot_pose[0], robot_pose[1], 'rx', markersize=15)
        r = pf._calc_repulsive_force(robot_pose[0:2])
        a = pf._calc_attractive_force(robot_pose[0:2])
        print(a)
        plt.quiver(robot_pose[0], robot_pose[1], r[0], -r[1], color='r')
        plt.quiver(robot_pose[0], robot_pose[1], a[0], -a[1], color='g')
        plt.grid(True)
        plt.pause(0.01)
        motion_vec = pf.get_force(robot_pose[0:2])
        robot.set_desired_motion(motion_vec)
