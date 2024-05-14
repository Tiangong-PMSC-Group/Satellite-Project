import numpy as np
import utilities as ut
from SatelliteState import SatelliteState
from config import config


class Radar():
    # Radar detection frequency from the configuration
    frequency = config['sim_config']['dt']['radar_freq']

    def __init__(self, position):
        """Creates a Radar object.

        Args: 
            position (numpy.array): starting position of the radar in Earth polar coordinates
        """
        # Initialize the radar with its position
        self.position = position
        # Initialize the last ping time to zero
        self.last_ping = 0
        # Variable to store the detected satellite position
        self.sate_pos_detected = None


    def line_of_sight(self, Sat_Pos, Earth_Eqn):
        """Check if the radar can see the Satellite.

        Args: 
            Sat_Pos (numpy.array): satellite position in Earth polar coordinates
            Earth_Eqn (Object Eath Function): objects function which returns point property

        Returns:
            bool: True if in LOS, False otherwise
        """
        #Build the iterative line towards satellite in Cartesian 
        x = ut.p_to_c(Sat_Pos)
        p = ut.p_to_c(self.position)

        v0 = x - p
        mag = np.sqrt(v0.dot(v0.T))
        v0 = v0 / mag

        #Checks if the line from dish to satellite goes into Earth.
        for t in range(10)[1:]:
            LOS_p = p + v0 * t
            if Earth_Eqn(LOS_p[0], LOS_p[1], LOS_p[2]) < 0:
                return False

        # If no LOS break, return True
        return True

        
    def is_time(self, current_time)-> SatelliteState:
        """Check the frequency. See if the radar can get the exact position of the satellite at this time.

        Args: 
            current_time (float): current value of time in the simutlation

        Returns:
            bool: True if enough time has passed, False otherwise
        """
        if current_time - self.last_ping > self.frequency:
            return True
        else:
            return False
        
    def detect_satellite_pos_short(self, Sat_Pos, current_time)-> SatelliteState:
        '''Get the position of the satellite and add Gaussian noise'''
        sate_pos_detected = self.add_noise(Sat_Pos)
        self.last_ping = current_time
        return SatelliteState(sate_pos_detected)


    def add_noise(self,position):
        """Adds Gaussian noise to a satellite position with different noise levels for each dimension.

        Args: 
            position (numpy.array): original position of the satellite

        Returns:
            numpy.array: noisy position according to noise_levels
        """
        if len(position) == 2:
            noise_levels = np.array([config['radar']['noise']['rho'],config['radar']['noise']['theta']])
        elif len(position) == 3:
            noise_levels = np.array([config['radar']['noise']['rho'],config['radar']['noise']['theta'],config['radar']['noise']['phi']])


        noisy_position = np.array(position) + np.random.normal(0, noise_levels, size=np.shape(position))
        return noisy_position
