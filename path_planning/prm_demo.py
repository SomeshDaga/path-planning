import configparser
import matplotlib.pyplot as plt
import numpy as np

from map import Map
from planners.prm import PRM
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

    # PRM Settings
    prm_num_samples = config.getint('PRM', 'num_samples')
    prm_num_neighbours = config.getint('PRM', 'num_neighbours')
    prm_inflation_radius = config.getfloat('PRM', 'inflation_radius')

    # Initialize map
    data = load_map(map_file)
    map = Map(data, resolution=map_resolution, d_theta=2*np.pi/num_scans)

    # Set start and goal positions
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])

    # Initialize algorithm
    prm = PRM(map, inflation_radius=prm_inflation_radius)

    # Initialize robot to starting position with zero heading
    robot = Robot(max_linear_vel=robot_max_linear_vel,
                  max_angular_vel=robot_max_angular_vel)
    robot.set_pose(start[0], start[1], 0.0)

    # map_inflated = map.get_inflated_obstacles_data(inflation_radius=inflation_radius)

    # Generate PRM samples
    prm.generate_roadmap(num_samples=prm_num_samples,
                         num_neighbours=prm_num_neighbours)
    vertices = prm.get_vertices()

    plt.figure()
    plt.imshow(map.get_data(), extent=[0, 10, 10, 0])
    for vertex in vertices:
        for n in vertex.get_neighbours():
            plt.plot([vertex.get_data()[0], n.get_data()[0]],
                     [vertex.get_data()[1], n.get_data()[1]],
                     color='green',
                     linestyle='solid',
                     marker='.',
                     markerfacecolor='blue')
    plt.show()

    plt.figure()
    for vertex in vertices:
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.imshow(map.get_data(), extent=[0, 10, 10, 0])
        plt.plot(vertex.get_data()[0], vertex.get_data()[1], '.r')
        for neighbour in vertex.get_neighbours():
            plt.plot(neighbour.get_data()[0], neighbour.get_data()[1], '.g')
        plt.grid(True)
        plt.pause(.0001)

