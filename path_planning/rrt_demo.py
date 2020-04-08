import configparser
import matplotlib.pyplot as plt
import numpy as np

from controller import Controller
from map import Map
from models.motion_models import diff_drive_1, bicycle_1
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
    robot_k_w = config.getfloat('ROBOT', 'k_w')
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
    start = np.array([0.5, 0.5])
    waypoints = np.array([[4.5, 5.0, 0],
                          [0.9, 9.0, np.pi/3],
                          [9.0, 1.0, np.pi/2],
                          [9.0, 9.0, np.pi]])

    # Initialize algorithm
    rrt = RRT(map, inflation_radius=rrt_inflation_radius)

    # Initialize robot to starting position with zero heading
    # and initialize its controller
    motion_model = diff_drive_1
    robot = Robot([start[0], start[1], 0.0],
                  motion_model)
    controller = Controller(robot,
                            k_w=robot_k_w,
                            max_linear_vel=robot_max_linear_vel,
                            max_angular_vel=robot_max_angular_vel)

    # Visit each waypoint sequentially
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Show the map
    plt.imshow(map.get_data(), extent=[0, 10, 10, 0])

    waypoint_idx = 0
    for waypoint in waypoints:
        waypoint_idx += 1
        path, tree = rrt.plan(robot.get_pose()[0:2],
                              waypoint[0:2],
                              iterations=rrt_num_iterations,
                              step=rrt_step_length)
        path_x = list(vertex.get_data()[0] for vertex in path)
        path_y = list(vertex.get_data()[1] for vertex in path)

        # Plot the RRT Tree
        rrt_lines = []
        for vertex in tree:
            for neighbour in vertex.get_neighbours():
                line, = ax.plot([vertex.get_data()[0], neighbour.get_data()[0]],
                                [vertex.get_data()[1], neighbour.get_data()[1]],
                                color='salmon',
                                linestyle='solid',
                                linewidth=1)
                rrt_lines.append(line)

        # Plot the path to the waypoint
        path_line, = ax.plot(path_x, path_y, color='green',
                             linestyle='dashed', linewidth=3, label='Planned Trajectory')

        # Set the path for the controller
        controller.set_path(path_x, path_y, waypoint[2])

        robot_marker = None
        dir_marker = None

        # Initialize an empty numpy array to store the robot trajectory
        # and add the start point to the trajectory
        robot_traj = np.empty((0, 3))
        robot_traj = np.append(robot_traj, [robot.get_pose()], axis=0)
        while not controller.finished():
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            # Apply control inputs and get resulting robot pose
            controller.loop(dt=dt)
            robot_pose = robot.get_pose()
            robot_traj = np.append(robot_traj, [robot_pose], axis=0)

            # Plot the robot position and direction of travel (remove previous markers)
            if robot_marker:
                ax.lines.remove(robot_marker)
            if dir_marker:
                dir_marker.remove()

            robot_marker, = ax.plot(robot_pose[0], robot_pose[1], 'bo',
                                    markersize=11.25, markeredgewidth=2, label='Robot')
            dir_marker = ax.quiver(robot_pose[0], robot_pose[1],
                                   np.cos(robot_pose[2]), -np.sin(robot_pose[2]),
                                   color='r', scale=2.5, scale_units='inches', label='Final Heading', zorder=4)
            plt.grid(True)
            plt.pause(.0001)

        skip = 60
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

        # Clear the RRT graph
        for line in rrt_lines:
            ax.lines.remove(line)

        # Clear the heading markers
        heading_markers.remove()

    print("Reached Goal!")
