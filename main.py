# Standart Libraries

import numpy as np



# Import files

import utilities
from Filter import ExtendedKalmanFilter
from Radar import Radar
from RadarSystem import RadarSystem
from Earth import Earth
from Satellite import Satellite
from SatelliteState import SatelliteState





# Initialize Earth
earth = Earth()


# Initialize RadarSystem
# Beijing Aerospace Command and Control Center

BACC =  RadarSystem(2, Earth()) 


# Initialize Satellite
R = earth.re/2 + 300000
theta = np.pi/2
phi = 0


angular_vel = 0.0010830807404
tang_vel = angular_vel * R
radial_velocity = 0
angular_velocity = 0


sat_state = SatelliteState(np.array([R, theta, phi]), np.array([0]), np.array([radial_velocity, tang_vel, angular_velocity]), np.array([0]))
tiagong = Satellite(sat_state, 0, earth=earth)

# Initialize the Kalman Filter
# tianhe is the chinese super computer

mean_0 = 0
cov_0 = 0
observation_matrix = 0
observation_noise = 0
process_noise = 0

tianhe = ExtendedKalmanFilter(mean_0, cov_0, earth, observation_matrix, observation_noise, process_noise)

# Run the simulation

# Number of simulation stepr
# Arbitrarialy large as it will stop at impact.
n = 10000000
simulation = tiagong.simulate(10000000)

sim_lenght = len(simulation.y[0])

predicted_states_satellite_cord = [mean_0]


for i in range(sim_lenght):

    current_state_satellite_cord = tiagong.get_position_at_t(i)
    current_state_earth_cord = utilities.spherical_to_spherical(current_state_satellite_cord)
    noise_states_earth_cord = BACC.try_detect_satellite(current_state_earth_cord, i) # Placeholder function. Change for real one
    

    if len(noise_states_earth_cord) > 0:
        for state_earth_cord in noise_states_earth_cord:
            state_satellite_cord = utilities.spherical_to_spherical(state_earth_cord)
            new_state_satellite_cord = tianhe.update(state_satellite_cord)
        
    new_state_satellite_cord = tianhe.forecast()

    predicted_states_satellite_cord += new_state_satellite_cord,

predicted_states_satellite_cord = np.array(predicted_states_satellite_cord)



