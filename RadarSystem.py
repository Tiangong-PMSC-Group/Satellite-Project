import math
import random

import numpy as np

from Interface.IRadarSystem import IRadarSystem
from SatelliteState import SatelliteState
from config import config
import utilities
from decorators import singleton
from Radar import Radar
from Earth import Earth


@singleton
class RadarSystem(IRadarSystem):
    radar_dt = config['sim_config']['dt']['radar_los']
    # earth_angular_velocity = config['earth']['angular_velocity']
    earth_angular_velocity = 2 * math.pi / 86400
    angular_change = radar_dt * earth_angular_velocity
    """A class control all radars to make them detect satellite positions periodically
     Predictor can get the informations by visiting radars list"""

    def __init__(self, counts, earth):
        self.earth = earth
        self.counts = counts
        self.init_radar_positions()

    def init_radar_positions(self):
        points = random_points_on_ellipse(Earth(), self.counts)
        # print(points) # check are these radar on the surface of the earth
        # c_pos = utilities.p_to_c(points[0])
        # print(Earth().ellipse_equation(c_pos[0], c_pos[1], c_pos[2]) )
        radars = []
        for point in points:
            radar = Radar(point)
            radars.append(radar)
        self.radars = radars

    def try_detect_satellite(self, sat_pos, current_time) -> SatelliteState:
        for radar in self.radars:
            find = radar.line_of_sight(sat_pos, self.earth.ellipse_equation)
            if find:
                radar.detect_satellite_pos(sat_pos, current_time)
            else:
                return None

    def update_radar_positions(self):
        for item in self.radars:
            item.position[1] = (item.position[1] + self.angular_change) % (2 * np.pi)
            # print(item.position)

def random_points_on_ellipse(self, num_points):
    points = []
    for _ in range(num_points):
        # Squared first eccentricity
        e2 = 1 - (self.rp ** 2 / self.re ** 2)

        # Generate random latitude and longitude
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)

        # Convert to radians
        phi = np.radians(latitude)
        lambda_ = np.radians(longitude)

        # Calculate the prime vertical radius
        N = self.re / np.sqrt(1 - e2 * np.sin(phi) ** 2)

        # Convert to geocentric Cartesian coordinates
        x = N * np.cos(phi) * np.cos(lambda_)
        y = N * np.cos(phi) * np.sin(lambda_)
        z = (N * (1 - e2)) * np.sin(phi)

        # Append the converted polar coordinates of the point
        points.append(utilities.c_to_p(np.array([x, y, z])))
    return points



'''
test code do not delete until last edition
'''
# pos = random_points_on_ellipse(Earth(), 1)[0]
# print("origin Cartesian:",pos)
# pos1 = utilities.p_to_c(pos)
# print("transformed to Cartesian:",pos1)
# print(Earth().ellipse_equation(pos1[0], pos1[1], pos1[2]))
# pos2 = utilities.c_to_p(pos1)
# print("transformed back to polar:",pos2)

# radar_system = RadarSystem(10, Earth())
# ''' Test code'''
# current_time = 1
# for i in range(1000):
#     radar_system.update_radar_positions()
#     radar_system.try_detect_satellite(np.array([100000000, math.pi, math.pi]), current_time)
#     current_time = current_time + 1
