
from abc import abstractmethod, ABCMeta

from SatelliteState import SatelliteState


class IRadarSystem(metaclass=ABCMeta):

    def __init__(self):
        pass

    '''
    For each timestep, simulate the position and return it.
    '''

    @abstractmethod
    def update_radar_positions(self):
        pass

    @abstractmethod
    def try_detect_satellite(self, sat_pos, current_time):
        pass



''' '''


