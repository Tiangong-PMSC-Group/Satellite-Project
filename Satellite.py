import numpy as np
from Filter import LinearKalmanFilter
from decorators import singleton

from scipy.integrate import solve_ivp

from config import config

import utilities

from SatelliteState import SatelliteState

from config import config 

from copy import deepcopy


#@singleton
class Satellite():
    """_summary_
    """
    def __init__(self, true_state0, prior_state, filter=None, earth=None):
        """_summary_

        Args:
            true_state0 (_type_): _description_
            prior_state (_type_): _description_
            filter (_type_, optional): _description_. Defaults to None.
            earth (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
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
            """_summary_
            """
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


    ####### Actual Dymanics

    def d_state(self, t, state):
        """_summary_

        Args:
            t (_type_): _description_
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
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
        """_summary_

        Args:
            r (_type_): _description_
            phi (_type_): _description_

        Returns:
            _type_: _description_
        """
        earth_coor = utilities.spherical_to_spherical([r, phi, self.plane_of_inclination])
        return self.earth.distance_to_surface(earth_coor)


    def simulate(self, time_limit):
        """_summary_

        Args:
            time_limit (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Run the full integration


        # Time span for the solution
        t_span = (0, time_limit*self.dt)  # From t=0 to t=10
        t_eval = np.linspace(t_span[0], t_span[1], int(time_limit/self.dt))  # Grid. Arbritarially large

        initial_state = self.true_state.get_state_sat_plane() #Just get position spherical orbit

        method = config['sim_config']['solver']

        sol = solve_ivp(self.d_state, t_span, initial_state, method = method,
                        t_eval=t_eval, events=self.crash_event)
        
        # Use later outside this class, wherever the program will run
        # Output results from the event
        #if sol.t_events[0].size > 0:
        #    print(f"Event occurred at t = {sol.t_events[0][0]}")
        #    print(f"State at event: r = {sol.y_events[0][0][0]}, v_r = {sol.y_events[0][0][1]}")

        self.solution = sol
        return sol
    
    def get_position_at_t(self, t):
        """_summary_

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        r = self.solution.y[0][t]
        theta = self.solution.y[2][t]
        phi = self.plane_of_inclination

        return np.array([r, theta, phi])

    #######


    
    def update_estimated_state(self, measurement=None):
        """_summary_

        Args:
            measurement (_type_, optional): _description_. Defaults to None.
        """
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


