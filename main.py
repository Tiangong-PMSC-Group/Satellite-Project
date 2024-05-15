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
    def __init__(self, n_radars = config['radar']['counts'], n = 10000000):

        self.n = n
        self.earth = Earth()
        self.BACC = RadarSystem(Earth(), n_radars)

        self.R = config['satellite']['initial_conditions']['distance']
        self.theta = config['satellite']['initial_conditions']['polar_angle']
        self.phi = config['satellite']['initial_conditions']['azimuthal_angle']


        self.angular_vel = config['satellite']['initial_conditions']['angular_velocity']
        self.tang_vel = self.angular_vel * self.R
        radial_velocity = config['satellite']['initial_conditions']['radial_velocity']
        azimuthal_velocity = config['satellite']['initial_conditions']['azimuthal_velocity']

        self.sat_state = SatelliteState(np.array([self.R, self.theta, self.phi]), np.array([0]), np.array([radial_velocity, self.tang_vel, azimuthal_velocity]), np.array([0]))
        self.tiangong = Satellite(self.sat_state, 0, earth=self.earth)



        # Initialize the Kalman Filter
        # tianhe is the chinese super computer

        # Adjust this to initialize at a random point with the same noise as radar
        initial_r = config['Kalman']['initial_r_guess']
        initial_angle = config['Kalman']['initial_angle_guess']
        self.mean_0 = np.array([initial_r , 0, 0, initial_angle, 0, 0])

        self.cov_0 = np.array(config['Kalman']['cov_matrix'])

        self.observation_noise = np.array(config['Kalman']['observation_noise'])

        self.Q = np.array(config['Kalman']['Q_matrix'])

        self.tianhe = ExtendedKalmanFilter(self.mean_0, self.cov_0, self.earth, observation_noise=self.observation_noise, process_noise=self.Q)

    
    def simulate(self):
        self.simulation = self.tiangong.simulate(10000000)

        self.R_simulation = self.simulation.y[0][:]
        self.rad_simulation = self.simulation.y[2][:]
        self.Phi_simulation = [np.pi/2 for i in self.rad_simulation]
    

    def predict(self):

        sim_lenght = len(self.simulation.y[0])

        m = deepcopy(self.mean_0)
        m = np.array([m[0], m[3]])
        predicted_states_satellite_cord = [m]
        radar_states_satellite_cord = [m]



        for i in range(int(sim_lenght)):
            
            if i < sim_lenght:
                current_state_satellite_cord = self.tiangong.get_position_at_t(i)
                #current_state_earth_cord = current_state_satellite_cord
                noise_states_satellite_cord = self.BACC.try_detect_satellite(current_state_satellite_cord, i)

                if len(noise_states_satellite_cord) > 0:
                    #print("Enter")
                    flag = 0
                    for state in noise_states_satellite_cord:

                        state_satellite_cord = state.pos
                        new_state_satellite_cord = self.tianhe.update(state_satellite_cord[:2])

                        if flag == 0:
                            #print( state_satellite_cord[:2])
                            radar_states_satellite_cord += state_satellite_cord[:2],
                            flag = 1

            forecast = self.tianhe.forecast()
            #print(f'Forecast: {forecast[0]}')
            #print(f'Simulation: {current_state_satellite_cord}')
            new_state_satellite_cord = [forecast[0][0][0], forecast[0][3][0]]

            predicted_states_satellite_cord += new_state_satellite_cord,


        self.R_predi, self.rad_predi = np.array(predicted_states_satellite_cord[:]).T
        self.R_radar, self.rad_radar = np.array(radar_states_satellite_cord[1:]).T



    def output(self):
        return [self.R_simulation, self.rad_simulation, self.R_predi, self.rad_predi, self.R_radar, self.rad_radar]
