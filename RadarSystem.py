import math
import threading
import time
import numpy as np

import utilities
from decorators import singleton
from Radar import Radar
from Earth import Earth


@singleton
class RadarSystem:
    """A class control all radars to make them detect satellite positions periodically
     Predictor can get the informations by visiting radars list"""

    def __init__(self, radars, earth):
        self.radars = radars
        self.earth = earth
        self.timer = None

    def try_detect_satellite(self, sat_pos, current_time):
        for radar in self.radars:
            find = radar.line_of_sight(sat_pos, self.earth.ellipse_equation)
            if find:
                radar.detect_satellite_pos(sat_pos, current_time)


def random_points_on_ellipse(self, num_points):
    points = []
    for _ in range(num_points):
        # generate latitude and longitude randomly
        latitude = np.random.uniform(-np.pi / 2, np.pi / 2)  # -π/2 到 π/2
        longitude = np.random.uniform(-np.pi, np.pi)  # -π 到 π

        # generate x y z using  latitude and longitude
        x = self.re * np.cos(latitude) * np.cos(longitude)
        y = self.re * np.cos(latitude) * np.sin(longitude)
        z = self.rp * np.sin(latitude)

        points.append(utilities.c_to_p(np.array([x, y, z])))
    return points


points = random_points_on_ellipse(Earth(), 10)
# print(points) # check are these radar on the surface of the earth
# c_pos = utilities.p_to_c(points[0])
# print(Earth().ellipse_equation(c_pos[0], c_pos[1], c_pos[2]) )
radars = []
for point in points:
    radar = Radar(point)
    radars.append(radar)
radar_system = RadarSystem(radars, Earth())

''' Test code'''
current_time = 1
for i in range(1000):
    radar_system.try_detect_satellite(np.array([1000000, math.pi, math.pi]), current_time)
    current_time = current_time + 1
