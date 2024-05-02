import numpy as np
from Filter import LinearKalmanFilter

class Satellite():
    """Satellite Class
    """
    def __init__(self, true_state0, filter=None, earth=None):


        self.filter = filter
        self.earth = earth

        self.true_state = true_state0
        self.estimated_state = self.filter.m #Initial guess stored in the Kalman Filter
        self.recorded_true_states = [self.true_state]
        self.recorded_estimated_states = [self.estimated_state]


    def update_true_state(self, dt):
        
        # Placeholder function, just as example
        # Change for the physical real one
        distance = self.earth.distane_to_surface(self.true_state) # Get distance to earth
        air = self.earth.air_density(distance)
        self.true_state = self.true_state + 1 * dt - air * dt # Dummy function
        self.recorded_states.append(self.true_state)

    
    def update_estimated_state(self, dt, measurement=None):
        
        distance = self.earth.distane_to_surface(self.estimated_state)
        air = self.earth.air_density(distance)
        transition_matrix = air*np.eye()*dt # Dummy function. Defines a new transition given the air
        
        if measurement == None:
            self.estimated_state = self.filter.forecast(transition_matrix)
        else:
            self.estimated_state = self.filter.update(measurement, transition_matrix)
        
        self.recorded_estimated_states.append(self.estimated_state)

    def full_simulation(self):
        # Implement a loop to predict to full trajectory, given the current state.
        # It has to initialize a new Kalman Filter.
        # Check if it reached the ground.
        # Else keep looping.
        # Return the full mean-trajectory and the uncertainty area on the ground.
        pass 
    


