import numpy as np
import math
import utilities as ut


class Radar():   
    def __init__(self, position):
        self.position = position
        self.last_ping = 0 
        self.sate_pos_detected = None
    
    # If the radar can detect the satellite, update sat position and last ping.
    def line_of_sight(self, Sat_Pos, Earth_Eqn, current_time):
        '''
        Sat - Satellite Position
        Earth - Earth Ellipse Equation
        '''

        #Build the iterative line towards satellite in Cartesian 
        x = ut.p_to_c(Sat_Pos)
        p = ut.p_to_c(self.position)

        v0 = x - p
        mag = np.sqrt(v0.dot(v0))
        v0 = v0/mag

        #Checks if the line from dish to satellite goes into Earth.
        for t in range(10)[1:]: #decrease if working for less comp time 
            LOS_p = p + v0*t
            if Earth_Eqn(LOS_p[0], LOS_p[1], LOS_p[2]) < 0:
                return False
        
        #If no LOS break, return True
        sate_pos_detected = add_noise(Sat_Pos)
        self.last_ping = current_time
        return True
        

def add_noise(position, noise_level=1.0):
    """
    Adds Gaussian noise to a satellite position.

    :param position: Original position of the satellite (numpy array or list).
    :param noise_level: Standard deviation of the Gaussian noise.
    :return: Noisy position.
    """
    noisy_position = np.array(position) + np.random.normal(0, noise_level, size=np.shape(position))
    return noisy_position

