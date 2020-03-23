import configparser

from .read_pgm import read_pgm

# Load configuration settings
config = configparser.ConfigParser()
config.read('config.ini')


def load_map():
    filename = config['MAP']['file']
    resolution = config.getfloat('MAP', 'resolution')
    return read_pgm(filename), resolution
