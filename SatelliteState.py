from dataclasses import dataclass
import utilities
import numpy as np

''' 
A dataclass which now is combination of time and position, and if the following steps need more data from the simulator,
add a field here is very convenient to let data pass through classes
'''


@dataclass
class SatelliteState:
    def __init__(self, pos: np.ndarray, cov: np.ndarray = None, velocity: np.ndarray = None, acceleration: np.ndarray = None):
        self.pos = pos
        self.cov = cov
        self.velocity = velocity
        self.acceleration = acceleration
    #current_time: int

    # Return the state in a single array
    def get_state(self):
        return np.concatenate([self.pos, self.velocity, self.acceleration])
    
    def get_state_sat_plane(self):
        new_pos = utilities.spherical_to_spherical(self.pos)
        return np.array([new_pos[0], self.velocity[0], new_pos[1], self.velocity[1]])

    def update_state(self, array):
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