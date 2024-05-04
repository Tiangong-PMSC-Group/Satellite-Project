import json
import math
import numpy as np

from Interface.IRadarSystem import IRadarSystem
from SatelliteState import SatelliteState
from config import config
import utilities
from decorators import singleton
from Earth import Earth
from TwoD.TwoDRadar import TwoDRadar
# @singleton
class TwoDRadarSystem(IRadarSystem):
    radar_dt = config['sim_config']['dt']['radar_los']
    # earth_angular_velocity = config['earth']['angular_velocity']
    earth_angular_velocity = 2 * math.pi / 86400
    angular_change = radar_dt * earth_angular_velocity

    def __init__(self, counts, earth):
        self.earth = earth
        self.counts = counts
        self.init_radar_positions()

    def init_radar_positions(self):
        points = random_points_on_circle_polar(self.counts,config['earth']['major_axis'])
        radars = []
        for point in points:
            radar = TwoDRadar(point)
            radars.append(radar)
        self.radars = radars

    def try_detect_satellite(self, sat_pos, current_time) -> SatelliteState:
        for radar in self.radars:
            find = radar.line_of_sight(sat_pos, self.earth.ellipse_equation)
            if find:
                radar.detect_satellite_pos(sat_pos, current_time)
                return sat_pos
            else:
                return None

    def update_radar_positions(self):
        for item in self.radars:
            item.position[1] = (item.position[1] + self.angular_change) % (2 * np.pi)
            # print(item.position)

def random_points_on_circle_polar(num_points, radius):
    points = []
    for _ in range(num_points):
        angle = np.random.uniform(0, 2 * np.pi)
        points.append(np.array([radius, angle]))
    return points


# radar_system = TwoDRadarSystem(10, Earth())
# ''' Test code'''
# current_time = 1
# for i in range(1000):
#     radar_system.update_radar_positions()
#     radar_system.try_detect_satellite(np.array([100000000, math.pi]), current_time)
#     current_time = current_time + 1

