import numpy as np
import math
import utilities as ut
from SatelliteState import SatelliteState
from config import config


class Radar():
    frequency = config['sim_config']['dt']['radar_freq']

    def __init__(self, position):
        self.position = position
        self.last_ping = 0
        self.sate_pos_detected = None

    # If the radar can detect the satellite, update sat position and last ping.
    def line_of_sight(self, Sat_Pos, Earth_Eqn):
        '''
        Sat - Satellite Position
        Earth - Earth Ellipse Equation
        '''

        #Build the iterative line towards satellite in Cartesian 
        x = ut.p_to_c(Sat_Pos)
        p = ut.p_to_c(self.position)

        v0 = x - p
        mag = np.sqrt(v0.dot(v0.T))
        v0 = v0 / mag

        #Checks if the line from dish to satellite goes into Earth.
        for t in range(10)[1:]:  #decrease if working for less comp time
            LOS_p = p + v0 * t
            if Earth_Eqn(LOS_p[0], LOS_p[1], LOS_p[2]) < 0:
                return False

        # If no LOS break, return True
        return True
    

    def detect_satellite_pos(self, Sat_Pos, current_time)-> SatelliteState:
        if current_time - self.last_ping > self.frequency:
            sate_pos_detected = self.add_noise(Sat_Pos)
            self.last_ping = current_time
            return SatelliteState(sate_pos_detected)
            # print(sate_pos_detected)
        else:
            return None
        
    def is_time(self, current_time)-> SatelliteState:
        if current_time - self.last_ping > self.frequency:
            self.last_ping = current_time
            return True
        else:
            return False
        
    def detect_satellite_pos_short(self, Sat_Pos, current_time)-> SatelliteState:
        sate_pos_detected = self.add_noise(Sat_Pos)
        self.last_ping = current_time
        return SatelliteState(sate_pos_detected)


    def add_noise(self,position):
        """
        Adds Gaussian noise to a satellite position with different noise levels for each dimension.

        :param position: Original position of the satellite (numpy array or list).
        :param noise_levels: List of standard deviations of the Gaussian noise for each dimension.
        :return: Noisy position.
        """
        if len(position) == 2:
            noise_levels = np.array([config['radar']['noise']['rho'],config['radar']['noise']['theta']])
        elif len(position) == 3:
            noise_levels = np.array([config['radar']['noise']['rho'],config['radar']['noise']['theta'],config['radar']['noise']['phi']])


        noisy_position = np.array(position) + np.random.normal(0, noise_levels, size=np.shape(position))
        return noisy_position
