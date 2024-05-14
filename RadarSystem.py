import math
import random
from typing import List

import numpy as np

from Interface.IRadarSystem import IRadarSystem
from SatelliteState import SatelliteState
from config import config
import utilities
from Radar import Radar
from Earth import Earth


"""A class control all radars to make them detect satellite positions periodically"""
class RadarSystem(IRadarSystem):

    radar_dt = config['sim_config']['dt']['main_dt']
    radar_los_frequency = config['sim_config']['dt']['radar_los']
    earth_angular_velocity = 2 * math.pi / 86400
    angular_change = radar_dt * earth_angular_velocity
    last_time = 0

    def __init__(self, earth, counts= None, seed=None, random_pos=True):
        self.earth = earth
        if counts is None:
            self.counts = config['radar']['counts']
        else:
            self.counts = counts
        if seed is not None:
            np.random.seed(seed)

        self.random_pos = random_pos
        self.init_radar_positions()

        
    ''' Initialize radar positions, supporting two modes: 
    evenly distributed on the equator or randomly distributed on the Earth's surface '''
    def init_radar_positions(self):
        if self.random_pos is True:
            points = utilities.random_points_on_ellipse(Earth(), self.counts)
        else:
            points = self.random_points_on_equator(Earth(), self.counts)
        radars = []
        for point in points:
            radar = Radar(point)
            radars.append(radar)
        self.radars = radars


    ''' Attempt to detect the satellite position '''
    def try_detect_satellite(self, sat_pos, current_time) -> list[SatelliteState]:
        results = []
        # The radar can only check whether the satellite is in it's detectable range at a certain frequency,
        # which is controlled by radar_los_frequency
        if current_time - self.last_time < self.radar_los_frequency:
            return results

        # Update radar positions due to Earth's rotation
        self.update_radar_positions(current_time - self.last_time)
        self.last_time = current_time

        ''' Even if the radar can check whether the satellite is in its detectable range, 
         it can only obtain the precise position of the satellite at a certain frequency.
         Here, we filter out those radars that can obtain the precise position of the satellite
         if it is within their detectable range'''
        radars = [self.radars[i] for i in range(len(self.radars)) if self.radars[i].is_time(current_time)]
        results = []

        # For each radar, first check if the satellite is within the detectable range
        for radar in radars:
            find = radar.line_of_sight(utilities.spherical_to_spherical(sat_pos), self.earth.ellipse_equation)
            # If within the detectable range, detect the precise position of the satellite
            if find:
                result = radar.detect_satellite_pos_short(sat_pos, current_time)
                results.append(result)
        return results

    ''' Update radar positions due to Earth's rotation '''
    def update_radar_positions(self,time_steps):
        for item in self.radars:
            new_state = item.position[2] + self.angular_change * time_steps
            item.position[2] = new_state % (2 * np.pi)

    '''Evenly distribute radar positions on the equator '''
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
            N = earth.re

            # Convert to geocentric Cartesian coordinates
            x = N * np.cos(phi) * np.cos(lambda_)
            y = N * np.cos(phi) * np.sin(lambda_)
            z = N * np.sin(phi)  

            # Append the converted polar coordinates of the point
            points.append(utilities.c_to_p(np.array([x, y, z])))

        return points

    '''Return radar positions '''
    def get_radar_positions(self):
        return [radar.position for radar in self.radars]


