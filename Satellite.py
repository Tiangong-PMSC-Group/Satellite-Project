import numpy as np
from Filter import LinearKalmanFilter
from decorators import singleton

from scipy.integrate import solve_ivp

from config import config
from ISimulator import ISimulator

import utilities

from SatelliteState import SatelliteState

from config import config 

from copy import deepcopy


#@singleton
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
        
        if filter is not None:
            self.filter = filter(prior_state.get_state(), prior_state.cov, None, observation_matrix, observation_noise, None)
        
        self.mass = config['satellite']['mass']
        self.area = config['satellite']['area']
        self.drag = config['satellite']['drag_coefficient']
        self.B = (self.area*self.drag)/self.mass
        
        self.earth = earth

        self.dt = config['sim_config']['dt']['main_dt']

        self.true_state = true_state0
        self.estimated_state = prior_state
        self.recorded_true_states = [deepcopy(self.true_state)]
        self.recorded_estimated_states = [deepcopy(self.estimated_state)]

        self.plane_of_inclination = self.true_state.pos[2]



        # Event class for handling the crash
        # For the solver
        class CrashEvent:
            def __init__(self, satellite):
                self.satellite = satellite
            
            def __call__(self, t, state):
                # Logic from the crash method
                r, v_r, phi, v_phi = state
                dist = self.satellite.plane_to_altitude(r, phi)['distance']
                if self.satellite.plane_to_altitude(r, phi)['inside']:
                    dist *= -1
                return dist
            
            terminal = True
            direction = -1

        self.crash_event = CrashEvent(self)

    ##### Old

    def update_true_state(self):
        # Update the state one step.
        
        # Placeholder function, just as example
        # Change for the physical real one
        
        distance = self.earth.distane_to_surface(self.true_state.pos)['distance'] # Get distance to earth
        air = self.earth.air_density(distance)
        self.true_state.pos = self.true_state.pos + self.true_state.velocity * self.dt 
        #- air * self.dt # Dummy function
        self.true_state.velocity = self.true_state.velocity + self.true_state.acceleration  * self.dt
        self.recorded_true_states.append(deepcopy(self.true_state))

    def run_simulation(self):
        # Run the whole simulation from "S_0" until the crash.
        # The intermediary result must be an array of arrays, each one in the satellite coordinates.
        # This vector should be passed through s_to_e and returned a new array of arrays in the earth coordinates
        pass

    #######


    ####### Actual Dymanics

    def d_state(self, t, state):
        # state in satelitte plane coordinates
        # "t" is not required in the actual calculations, but it is neeeded
        # for the scipy solver
        # r -> Radial Distance
        # v_r -> Radial Velocity
        # phi -> Angle (axis of rotation of the plane)
        # v_phi -> TANGENCIAL velocity (not angular velocity)

        r, v_r, phi, v_phi = state

        Me = self.earth.mass
        G = self.earth.grav_const
        B = self.B

        #rho = self.earth.air_density(plane_to_altitude( r, phi))
        rho = 0.5


        d_r = v_r
        d_vr = - (G*Me/r**2) + (v_phi**2)/r
        d_phi = v_phi/r
        d_v_phi = -0.5*self.earth.air_density(self.plane_to_altitude(r,phi)['distance']) * v_phi**2 * B + (v_phi*v_r)/r

        return np.array([d_r, d_vr, d_phi, d_v_phi])

    def plane_to_altitude(self, r, phi):
        earth_coor = utilities.satellite_to_earth([r, phi, self.plane_of_inclination])
        return self.earth.distance_to_surface(earth_coor)


    #def crash(self, t, state):
     #   #Check if satellite hitted the earth
      #  r, v_r, phi, v_phi = state
       # return self.plane_to_altitude(r, phi)['distance']
    
    


    def simulate(self, time_limit):
        # Run the full integration


        # Time span for the solution
        t_span = (0, time_limit)  # From t=0 to t=10
        t_eval = np.linspace(t_span[0], t_span[1], time_limit*self.dt)  # Grid. Arbritarially large

        initial_state = self.true_state.get_state_sat_plane() #Just get position

        method = config['sim_config']['solver']

        sol = solve_ivp(self.d_state, t_span, initial_state, method = method,
                        t_eval=t_eval, events=self.crash_event)
        
        # Use later outside this class, wherever the program will run
        # Output results from the event
        #if sol.t_events[0].size > 0:
        #    print(f"Event occurred at t = {sol.t_events[0][0]}")
        #    print(f"State at event: r = {sol.y_events[0][0][0]}, v_r = {sol.y_events[0][0][1]}")

        return sol

    #######


    
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

    def full_prediction(self):
        # Implement a loop to predict to full trajectory, given the current state.
        # It has to initialize a new Kalman Filter.
        # Check if it reached the ground.
        # Else keep looping.
        # Return the full mean-trajectory and the uncertainty area on the ground.
        # It is a PREDICTION, not the true simulation. Use for the bonus part 
        pass 


