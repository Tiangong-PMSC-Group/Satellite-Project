import math
import threading
import time
import numpy as np
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

    def try_detect_satellite(self, sat_pos,current_time):
        for radar in self.radars:
            find = radar.line_of_sight(sat_pos, self.earth.ellipse_equation)
            if find:
                radar.detect_satellite_pos(sat_pos, current_time)


""" TODO :
Init the positions of the radors, they should be on the surface of the earth.
"""
radars = [Radar(np.array([[0], [0], [0]])), Radar(np.array([[0], [0], [0]]))]
radar_system = RadarSystem(radars, Earth())


''' Test code'''
current_time = 1
for i in range(1000):
    radar_system.try_detect_satellite(np.array([[1000000], [math.pi], [math.pi]]), current_time)
    current_time = current_time + 1

