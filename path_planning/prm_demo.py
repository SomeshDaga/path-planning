import configparser
import matplotlib.pyplot as plt
import numpy as np

from controller import Controller
from map import Map
from models.motion_models import diff_drive_1
from planners import PRM
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
    robot_k_w = config.getfloat('ROBOT', 'k_w')
    robot_max_linear_vel = config.getfloat('ROBOT', 'max_linear_vel')
    robot_max_angular_vel = config.getfloat('ROBOT', 'max_angular_vel')

    # PRM Settings
    prm_num_samples = config.getint('PRM', 'num_samples')
    prm_num_neighbours = config.getint('PRM', 'num_neighbours')
    prm_inflation_radius = config.getfloat('PRM', 'inflation_radius')

    # Initialize map
    data = load_map(map_file)
    map = Map(data, resolution=map_resolution, d_theta=2*np.pi/num_scans)

    # Set start position and waypoints
    start = np.array([0.5, 0.5])
    waypoints = np.array([[7.0, 1.5, 3*np.pi/2],
                          [9.0, 5.0, np.pi],
                          [3.0, 9.5, np.pi/2],
                          [0.5, 5.0, 0]])

    # Initialize algorithm
    prm = PRM(map, inflation_radius=prm_inflation_radius)

    # Initialize robot to starting position with zero heading
    # and initialize its controller
    motion_model = diff_drive_1
    robot = Robot([start[0], start[1], 0.0], diff_drive_1)
    controller = Controller(robot,
                            k_w=robot_k_w,
                            max_linear_vel=robot_max_linear_vel,
                            max_angular_vel=robot_max_angular_vel)

    # Initialize an empty numpy array to store the robot trajectory
    # and add the start point to the trajectory
    robot_traj = np.empty((0, 3))
    robot_traj = np.append(robot_traj, [robot.get_pose()], axis=0)

    # Generate PRM samples
    prm.generate_roadmap(num_samples=prm_num_samples,
                         num_neighbours=prm_num_neighbours)

    # Show the generated roadmap
    vertices = prm.get_vertices()
    plt.figure()
    plt.imshow(map.get_data(), extent=[0, 10, 10, 0])
    print("This might take up to 45 seconds...hold on")
    for vertex in vertices:
        for neighbour in vertex.get_neighbours():
            plt.plot([vertex.get_data()[0], neighbour.get_data()[0]],
                     [vertex.get_data()[1], neighbour.get_data()[1]],
                     color='green', linestyle='solid', marker='.',
                     markeredgecolor='blue', linewidth=1)
    plt.draw()
    print('Press spacebar to continue...')
    plt.waitforbuttonpress(timeout=1000)

    # Visit each waypoint sequentially
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(map.get_data(), extent=[0, 10, 10, 0])
    waypoint_idx = 0
    for waypoint in waypoints:
        waypoint_idx += 1
        path = prm.find_shortest_path(robot.get_pose()[0:2],
                                      waypoint[0:2],
                                      num_neighbours=prm_num_neighbours)
        path_x = list(vertex.get_data()[0] for vertex in path)
        path_y = list(vertex.get_data()[1] for vertex in path)

        # Set the path for the controller
        controller.set_path(path_x, path_y, waypoint[2])

        # Draw the path
        path_line, = ax.plot(path_x, path_y, color='green', linestyle='dashed', linewidth=3, label='Planned Trajectory')

        robot_marker = None
        dir_marker = None
        while not controller.finished():
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            # Apply control inputs and get resulting robot pose
            controller.loop(dt=dt)
            robot_pose = robot.get_pose()
            robot_traj = np.append(robot_traj, [robot_pose], axis=0)

            if robot_marker:
                ax.lines.remove(robot_marker)
            if dir_marker:
                dir_marker.remove()

            robot_marker, = ax.plot(robot_pose[0], robot_pose[1], 'bo',
                                    markersize=11.25, markeredgewidth=2, label='Robot')
            dir_marker = ax.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[2]), -np.sin(robot_pose[2]),
                                   color='r', scale=2.5, scale_units='inches', zorder=4,
                                   label='Final Heading')
            plt.grid(True)
            plt.pause(.0001)

        skip = 30
        plt.annotate('{0}'.format(waypoint_idx), xy=(waypoint[0], waypoint[1]),
                     textcoords='offset points', xytext=(3, 7),
                     fontweight='bold', fontsize='14', color='indianred')
        actual_traj_line, = ax.plot(robot_traj[:, 0], robot_traj[:, 1], color='black', label='Actual Trajectory')
        heading_markers = ax.quiver(robot_traj[::skip, 0], robot_traj[::skip, 1],
                                    np.cos(robot_traj[::skip, 2]), -np.sin(robot_traj[::skip, 2]),
                                    color='blue', units='inches', scale=4.0, zorder=3, label='Heading')
        plt.legend(handles=[robot_marker, dir_marker, path_line, actual_traj_line, heading_markers],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        plt.draw()
        print("Press spacebar to continue...")
        plt.waitforbuttonpress(timeout=1000)

    print("Reached Goal!")
