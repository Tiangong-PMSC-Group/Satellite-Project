# Standart Libraries

import numpy as np
from copy import deepcopy

# Import files

import utilities
from Filter import ExtendedKalmanFilter
from Radar import Radar
from RadarSystem import RadarSystem
from Earth import Earth
from Satellite import Satellite
from SatelliteState import SatelliteState

from config import config


class Main():
    def __init__(self, n_radars, n = 10000000):

        self.n = n

        self.earth = Earth()
        self.BACC = RadarSystem(Earth(), n_radars)

        self.R = config['satellite']['initial_conditions']['distance']
        self.theta = config['satellite']['initial_conditions']['polar_angle']
        self.phi = config['satellite']['initial_conditions']['azimuthal_angle']


        self.angular_vel = 0.0011313 # Send to config
        self.tang_vel = self.angular_vel * self.R
        radial_velocity = 0
        azimuthal_velocity = 0

        self.sat_state = SatelliteState(np.array([self.R, self.theta, self.phi]), np.array([0]), np.array([radial_velocity, self.tang_vel, azimuthal_velocity]), np.array([0]))
        self.tiagong = Satellite(self.sat_state, 0, earth=self.earth)



        # Initialize the Kalman Filter
        # tianhe is the chinese super computer

        # Send those things to the config
        r_noise = config['radar']['noise']['rho']
        t_noise = config['radar']['noise']['theta']
        # Adjust this to initialize at a random point with the same noise as radar
        self.mean_0 = np.array([self.R+100000, 0, 0, np.pi-3, 0, 0])

        # cov_0 = np.array([
        #     [3.98e8, 0, 0, 0, 0, 0],
        #     [0, 1.092e1, 0, 0, 0, 0],
        #     [0, 0, 1e0, 0, 0, 0],
        #     [0, 0, 0, 1.9533, 0, 0],
        #     [0, 0, 0, 0, 5.194e-1, 0],
        #     [0, 0, 0, 0, 0, 8.03e-2]
        # ])

        self.cov_0 = np.array([
            [1e4, 0, 0, 0, 0, 0],
            [0, 1e1, 0, 0, 0, 0],
            [0, 0, 1e0, 0, 0, 0],
            [0, 0, 0, 1e1, 0, 0],
            [0, 0, 0, 0, 1e0, 0],
            [0, 0, 0, 0, 0, 1e-2]
        ])

        self.observation_noise = np.array([[2e2, 0],
                    [0, 1e-1]])

        self.Q = np.array([
            [100, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0.1, 0, 0, 0],
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0, 0.02, 0],
            [0, 0, 0, 0, 0, 0.002]
        ])

        self.tianhe = ExtendedKalmanFilter(self.mean_0, self.cov_0, self.earth, observation_noise=self.observation_noise, process_noise=self.Q)

    
    def simulate(self):
        self.simulation = self.tiagong.simulate(10000000)

        self.R_simulation = self.simulation.y[0][:]
        self.rad_simulation = self.simulation.y[2][:]
        self.Phi_simulation = [np.pi/2 for i in self.rad_simulation]
    

    def prediction(self):

        sim_lenght = len(self.simulation.y[0])

        m = deepcopy(self.mean_0)
        m = np.array([m[0], m[3]])
        predicted_states_satellite_cord = [m]
        radar_states_satellite_cord = [m]



        for i in range(int(sim_lenght)):
            
            if i < sim_lenght:
                current_state_satellite_cord = self.tiagong.get_position_at_t(i)
                #current_state_earth_cord = utilities.spherical_to_spherical(current_state_satellite_cord)
                current_state_earth_cord = current_state_satellite_cord
                noise_states_earth_cord = self.BACC.try_detect_satellite(current_state_earth_cord, i)
                #if len(noise_states_earth_cord) > 0:
                 #   print(current_state_earth_cord, noise_states_earth_cord[0].pos, self.rad_simulation[i])

                #print(len(noise_states_earth_cord))
                if len(noise_states_earth_cord) > 0:
                    #print("Enter")
                    flag = 0
                    for state_earth_cord in noise_states_earth_cord:
                        #print("Update0")
                        #state_satellite_cord = utilities.spherical_to_spherical(state_earth_cord.pos)
                        state_satellite_cord = state_earth_cord.pos
                        new_state_satellite_cord = self.tianhe.update(state_satellite_cord[:2])

                        if flag == 0:
                            #print( state_satellite_cord[:2])
                            radar_states_satellite_cord += state_satellite_cord[:2],
                            flag = 1
                    
            forecast = self.tianhe.forecast()
            new_state_satellite_cord = [forecast[0][0][0], forecast[0][3][0]]

            predicted_states_satellite_cord += new_state_satellite_cord,


        self.R_predi, self.rad_predi = np.array(predicted_states_satellite_cord[:]).T
        self.R_radar, self.rad_radar = np.array(radar_states_satellite_cord[1:]).T



    def output(self):
        return [self.R_simulation, self.rad_simulation, self.R_predi, self.rad_predi, self.R_radar, self.rad_radar]
