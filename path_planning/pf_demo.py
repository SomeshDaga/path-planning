import configparser
import matplotlib.pyplot as plt
import numpy as np

from controller import VectorController
from map import Map
from models.motion_models import diff_drive_1, diff_drive_2
from planners import PotentialField
from robot import Robot
from utils import load_map

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
    robot_k_w = config.getfloat('ROBOT', 'k_w')
    robot_max_linear_vel = config.getfloat('ROBOT', 'max_linear_vel')
    robot_max_angular_vel = config.getfloat('ROBOT', 'max_angular_vel')

    # Potential Field Settings
    k_a = config.getfloat('PF', 'k_a')
    k_r = config.getfloat('PF', 'k_r')
    obstacle_dist_thresh = config.getfloat('PF', 'obstacle_dist_thresh')

    # Initialize map
    data = load_map(map_file)
    map = Map(data, resolution=map_resolution, d_theta=2*np.pi/num_scans)

    # Set start and goal positions
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])

    # Initialize algorithm
    pf = PotentialField(goal, map, k_a=k_a, k_r=k_r, obs_dist_thresh=obstacle_dist_thresh)

    # Initialize robot to starting position with zero heading
    motion_model = diff_drive_1
    robot = Robot([start[0], start[1], 0.0], motion_model)
    controller = VectorController(robot,
                                  k_w=robot_k_w,
                                  max_linear_vel=robot_max_linear_vel,
                                  max_angular_vel=robot_max_angular_vel)

    # Initialize an empty numpy array to store the robot trajectory
    # and add the start point to the trajectory
    robot_traj = np.empty((0, 3))
    robot_traj = np.append(robot_traj, [robot.get_pose()], axis=0)

    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    a_marker = None
    r_marker = None
    while not robot.is_at(goal, tol=1e-2):
        ax.cla()

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        robot_pose = robot.get_pose()
        robot_traj = np.append(robot_traj, [robot_pose], axis=0)

        # Plot the live obstacles being detected on the map
        map_image = map.get_data()
        obstacles, distances = map.get_obstacles_in_radius(robot_pose[0:2], 5)
        for obs, d in zip(obstacles, distances):
            obs = map.worldToMap(obs)
            map_image[obs[1], obs[0]] = 128
        ax.imshow(map_image, extent=[0, 10, 10, 0])

        # Plot robot position
        ax.plot(robot_pose[0], robot_pose[1], 'ro', label='Robot', markersize=11.25)

        # Plot directions of attractive and repulsive forces
        r = pf.calc_repulsive_force(robot_pose[0:2])
        # Only normalize the force if it is non-zero
        if r.any():
            r = r / np.linalg.norm(r)  # Normalize the repulsive force to only plot direction

        a = pf.calc_attractive_force(robot_pose[0:2])
        a = a / np.linalg.norm(a)  # Normalize the attractive force to only plot direction
        a_marker = \
            ax.quiver(robot_pose[0], robot_pose[1], a[0], -a[1], color='g', units='inches', scale=2.0)
        if r.any():
            r_marker = \
                ax.quiver(robot_pose[0], robot_pose[1], r[0], -r[1], color='r', units='inches', scale=2.0)
        else:
            r_marker = None

        # Pause animation between frames
        plt.grid(True)
        plt.pause(0.01)

        # Get the motion vectors to apply to the robot for propagation
        # in the next update cycle
        motion_vec = pf.get_force(robot_pose[0:2])
        # Propagate robot using motion vector and get its pose
        controller.loop(motion_vec=motion_vec, dt=dt)
        # robot.set_desired_motion(motion_vec)

    # Successfully Reached Goal
    print("Reached Goal!")

    # Remove force markers
    if a_marker:
        a_marker.remove()
    if r_marker:
        r_marker.remove()

    # Plot the robot trajectory, including headings
    ax.plot(robot_traj[:, 0], robot_traj[:, 1], '-r',
            linewidth=2, label='Trajectory', zorder=2)
    skip = 20
    ax.quiver(robot_traj[::skip, 0], robot_traj[::skip, 1],
              np.cos(robot_traj[::skip, 2]), -np.sin(robot_traj[::skip, 2]),
              color='b', units='inches', label='Heading', scale=4.0, zorder=3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.draw()
    plt.show()
