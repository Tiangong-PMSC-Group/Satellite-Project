import math
import random
from typing import List

import numpy as np

from Interface.IRadarSystem import IRadarSystem
from SatelliteState import SatelliteState
from config import config
import utilities
from decorators import singleton
from Radar import Radar
from Earth import Earth


# @singleton
class RadarSystem(IRadarSystem):
    radar_dt = config['sim_config']['dt']['main_dt']
    earth_angular_velocity = 2 * math.pi / 86400
    angular_change = radar_dt * earth_angular_velocity
    last_time = 0
    """A class control all radars to make them detect satellite positions periodically
     Predictor can get the informations by visiting radars list"""

    def __init__(self, counts, earth, seed=None):
        self.earth = earth
        self.counts = counts

        if seed is not None:
            np.random.seed(seed)

            
        self.init_radar_positions()

        

    def init_radar_positions(self):
        points = utilities.random_points_on_ellipse(Earth(), self.counts)
        # points = self.random_points_on_equator(Earth(), self.counts)
        radars = []
        for point in points:
            radar = Radar(point)
            radars.append(radar)
        self.radars = radars


#    def try_detect_satellite(self, sat_pos, current_time) -> list[SatelliteState]:
#        self.update_radar_positions(current_time - self.last_time)
#        self.last_time = current_time
#        results = []
#        for radar in self.radars:
#            find = radar.line_of_sight(sat_pos, self.earth.ellipse_equation)
#             if find:
#                 result = radar.detect_satellite_pos(sat_pos, current_time)
#                 if result is not None:
#                     results.append(result)
#         return results

    def try_detect_satellite(self, sat_pos, current_time) -> list[SatelliteState]:
        self.update_radar_positions(current_time - self.last_time)
        self.last_time = current_time
        radars = [self.radars[i] for i in range(len(self.radars)) if self.radars[i].is_time(current_time)]
        results = []
        for radar in radars:
            find = radar.line_of_sight(sat_pos, self.earth.ellipse_equation)
            if find:
                result = radar.detect_satellite_pos_short(sat_pos, current_time)
                results.append(result)
        return results

    def update_radar_positions(self,time_steps):
        for item in self.radars:
            new_state = item.position[2] + self.angular_change * time_steps
            item.position[2] = new_state % (2 * np.pi)


    def random_points_on_equator(self,earth, num_points):
        points = []
        # Interval between points in degrees
        interval = 360 / num_points

        for i in range(num_points):
            # Longitude at equal intervals
            longitude = i * interval

            # Latitude is 0 for the equator
            latitude = 0

            # Convert to radians
            phi = np.radians(latitude)
            lambda_ = np.radians(longitude)

            # Calculate the prime vertical radius, N, for latitude = 0 simplifies the formula
            N = earth.re  # At the equator, the simplification occurs because sin(0) = 0

            # Convert to geocentric Cartesian coordinates
            x = N * np.cos(phi) * np.cos(lambda_)
            y = N * np.cos(phi) * np.sin(lambda_)
            z = N * np.sin(phi)  # This will be zero at the equator

            # Append the converted polar coordinates of the point
            points.append(utilities.c_to_p(np.array([x, y, z])))

        return points



'''
test code do not delete until last edition
'''

# pos = random_points_on_equator(Earth(), 2)[0]
# print("origin Cartesian:",pos)
# print("origin Cartesian:",random_points_on_equator(Earth(), 2)[1])
# pos1 = utilities.p_to_c(pos)
# pos3 = random_points_on_equator(Earth(), 2)[1]
# print("transformed to Cartesian:",pos1)
# print("transformed to Cartesian:",utilities.p_to_c(pos3))
# print(Earth().ellipse_equation(pos1[0], pos1[1], pos1[2]))
# pos2 = utilities.c_to_p(pos1)
# print("transformed back to polar:",pos2)

# radar_system = RadarSystem(10, Earth())
# print(radar_system)
# radar_system = RadarSystem(1, Earth())
# print(radar_system)
# ''' Test code'''
# current_time = 1
# for i in range(1000):
#     radar_system.update_radar_positions()
#     radar_system.try_detect_satellite(np.array([100000000, math.pi, math.pi]), current_time)
#     current_time = current_time + 1
