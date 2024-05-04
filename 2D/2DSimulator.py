import numpy as np
import matplotlib.pyplot as plt

import utilities
from ISimulator import ISimulator
from SatelliteState import SatelliteState
import json
'''
"A 2D simulator without considering atmospheric drag, suitable for early project testing and integration."
'''
class Two_D_Simulator(ISimulator):
    """Simulator for satellite positions in a two-dimensional orbital plane."""

    def __init__(self):
        super().__init__()
        self.G = config['sim_config']['gravitational_constant'] # Gravitational constant in m^3 kg^-1 s^-2
        self.M = config['earth']['mass']  # Mass of Earth in kg
        self.dt = config['sim_config']['dt']['main_dt']  # Timestep in seconds
        self.r = config['earth']['major_axis'] + 400e3  # Initial distance from Earth's center plus 400 km
        self.theta = 0  # Initial angle
        self.dr = 0  # Initial radial velocity
        self.dtheta = np.sqrt(self.G * self.M / self.r ** 3)  # Initial angular velocity, assuming circular orbit

    def simulate(self) -> SatelliteState:
        """Simulates one timestep and returns the satellite position and velocity."""

        # Calculate new position using Forward Euler Method
        new_r = self.r + self.dt * self.dr
        new_theta = self.theta + self.dt * self.dtheta

        # Calculate new velocity
        new_dr = self.dr - self.dt * (self.r * self.dtheta ** 2 - self.G * self.M / self.r ** 2)
        new_dtheta = self.dtheta + self.dt * (2 * self.dr * self.dtheta / self.r)

        # Update current state
        self.r, self.theta, self.dr, self.dtheta = new_r, new_theta, new_dr, new_dtheta

        return SatelliteState(velocity=np.array([self.dr, self.dtheta]), pos=np.array([self.r, self.theta]),
                              current_time=0)
def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

config = load_config('../config.json')

def print_trajactory_for_test():
    # Usage of Two_D_Simulator to simulate an orbit for a day
    simulator = Two_D_Simulator()
    positions = []

    num_steps = int(3600 * 24 / simulator.dt)
    for _ in range(num_steps):
        satelliteState = simulator.simulate()
        positions.append(utilities.p_to_c(satelliteState.pos))

    # Convert positions list to numpy array for plotting
    positions = np.array(positions)
    plt.figure(figsize=(8, 8))
    plt.plot(positions[:, 0], positions[:, 1])
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Orbit Simulation using Two_D_Simulator')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

print_trajactory_for_test()