import numpy as np
from Filter import LinearKalmanFilter
from decorators import singleton

from config import config
from ISimulator import ISimulator

from SatelliteState import SatelliteState

from copy import deepcopy


@singleton
class Satellite(ISimulator):
    """Satellite Class
    """
    def __init__(self, true_state0, prior_state, filter=None, earth=None):
        # true_state is a SatelliteState data type

        # Initialize the Kalman Filter
        # Create a zero matrix of size n x 3n for the observation matrix
        n = len(true_state0.pos)
        observation_matrix = np.fill_diagonal(np.zeros((n, 3 * n)), 1)
        observation_noise = np.diag(list(config['radar']['noise'].values())) #Check if the noise values in the radar (in the config file) are in the correct order
        self.filter = filter(prior_state.get_state(), prior_state.cov, None, observation_matrix, observation_noise, None)
        
        
        self.earth = earth

        self.dt = config['sim_config']['dt']['main_dt']

        self.true_state = true_state0
        self.estimated_state = prior_state
        self.recorded_true_states = [deepcopy(self.true_state)]
        self.recorded_estimated_states = [deepcopy(self.estimated_state)]


    def update_true_state(self):
        
        # Placeholder function, just as example
        # Change for the physical real one
        
        distance = self.earth.distane_to_surface(self.true_state.pos)['distance'] # Get distance to earth
        air = self.earth.air_density(distance)
        self.true_state.pos = self.true_state.pos + self.true_state.velocity * self.dt 
        #- air * self.dt # Dummy function
        self.true_state.velocity = self.true_state.velocity + self.true_state.acceleration  * self.dt
        self.recorded_true_states.append(deepcopy(self.true_state))

    
    def update_estimated_state(self, measurement=None):
        
        distance = self.earth.distane_to_surface(self.estimated_state.pos)['distance']
        air = self.earth.air_density(distance)
        transition_matrix = np.eye(9) # Dummy function. Defines a new transition given the air
        
        if measurement is None:
            m, c = self.filter.forecast(transition_matrix)
            self.estimated_state.update_state(m)
            self.estimated_state.cov = c
        else:
            m, c = self.filter.update(measurement, transition_matrix)
            self.estimated_state.update_state(m)
            self.estimated_state.cov = c
                    
        self.recorded_estimated_states.append(deepcopy(self.estimated_state))

    def full_simulation(self):
        # Implement a loop to predict to full trajectory, given the current state.
        # It has to initialize a new Kalman Filter.
        # Check if it reached the ground.
        # Else keep looping.
        # Return the full mean-trajectory and the uncertainty area on the ground.
        # It is a PREDICTION, not the true simulation. Use for the bonus part 
        pass 
    


