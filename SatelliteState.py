from dataclasses import dataclass
import utilities
import numpy as np

''' 
A dataclass for Satellite information. A convenient way to store and acess data.
'''


@dataclass
class SatelliteState:
    def __init__(self, pos: np.ndarray, cov: np.ndarray = None, velocity: np.ndarray = None, acceleration: np.ndarray = None):
        """_summary_

        Args:
            pos (np.ndarray): _description_
            cov (np.ndarray, optional): _description_. Defaults to None.
            velocity (np.ndarray, optional): _description_. Defaults to None.
            acceleration (np.ndarray, optional): _description_. Defaults to None.
        """
        self.pos = pos #in earths spherical system
        self.cov = cov
        self.velocity = velocity
        self.acceleration = acceleration

    # Return the state in a single array
    def get_state(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return np.concatenate([self.pos, self.velocity, self.acceleration])
    
    def get_state_sat_plane(self):
        """

        Args:
            state (np.array()): Spherical coordinates earth

        Returns:
            np.array([R, Polar, Azimuthal]): Spherical coordinates satellite
        """
        new_pos = utilities.spherical_to_spherical(self.pos)
        return np.array([new_pos[0], self.velocity[0], new_pos[1], self.velocity[1]])

    def update_state(self, array):
        """_summary_

        Args:
            array (_type_): _description_

        Raises:
            ValueError: _description_
        """
        # Assuming the size of each component (pos, velocity, acceleration) is known and fixed
        pos_size = len(self.pos)
        velocity_size = len(self.velocity)
        acceleration_size = len(self.acceleration)

        # Validate the input array length
        if len(array) != (pos_size + velocity_size + acceleration_size):
            raise ValueError("The length of the input array does not match the expected total length of state components.")

        # Split and update the state components
        self.pos = array[:pos_size]
        self.velocity = array[pos_size:pos_size + velocity_size]
        self.acceleration = array[pos_size + velocity_size:]