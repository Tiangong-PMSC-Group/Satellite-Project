import numpy as np
import matplotlib.pyplot as plt
from config import config
import utilities
from ISimulator import ISimulator
from SatelliteState import SatelliteState
import pickle
from ISimulator import ISimulator, FakeSimulator, RealSimulator
class TwoDSimulator(ISimulator):
    def __init__(self):
        super().__init__()

        self.r = config['earth']['major_axis'] + 400e3  # Initial altitude 4000 km above Earth's surface
        self.theta = 0
        self.dr = 0
        self.dtheta = np.sqrt(0.8 * self.G * self.M / self.r ** 3)



    def simulate(self, current_time) -> SatelliteState:
        # Current speed
        v = np.sqrt(self.dr ** 2 + (self.r * self.dtheta) ** 2)

        # Atmospheric density
        if self.r > config['earth']['major_axis']:
            altitude = self.r - config['earth']['major_axis']
            rho = 1.225 * np.exp(-altitude / 8500)  # Density decreases with altitude
        else:
            rho = 0  # Below the surface of the Earth, no atmosphere

        # Drag force calculation
        Cd = config['satellite']['drag_coefficient']  # Drag coefficient for a spherical object
        A = config['satellite']['area']  # Cross-sectional area of the satellite
        Fd = -0.5 * Cd * rho * v ** 2 * A
        ad = Fd / config['satellite']['mass']  # Acceleration due to drag

        # Update dynamics with drag
        new_dr = self.dr - self.dt * (self.r * self.dtheta ** 2 - self.G * self.M / self.r ** 2) + self.dt * ad
        new_dtheta = self.dtheta + self.dt * (2 * self.dr * self.dtheta / self.r)

        # position update
        new_r = self.r + self.dt * new_dr
        new_theta = self.theta + self.dt * new_dtheta

        # Update state
        self.r, self.theta, self.dr, self.dtheta = new_r, new_theta, new_dr, new_dtheta

        if self.r < config['earth']['major_axis']:  # Check if the satellite has crashed
            return None
        return SatelliteState(velocity=np.array([self.dr, self.dtheta]), pos=np.array([self.r, self.theta]),
                              current_time=current_time)

 # round trajectory of not falling satellite
    def simulate_not_fall(self, current_time) -> SatelliteState:
        new_r = self.r + self.dt * self.dr
        new_theta = self.theta + self.dt * self.dtheta
        new_dr = self.dr - self.dt * (self.r * self.dtheta ** 2 - self.G * self.M / self.r ** 2)
        new_dtheta = self.dtheta + self.dt * (2 * self.dr * self.dtheta / self.r)
        self.r, self.theta, self.dr, self.dtheta = new_r, new_theta, new_dr, new_dtheta

        if self.r < config['earth']['major_axis']:  # Check if the satellite has crashed
            return None
        return SatelliteState(velocity=np.array([self.dr, self.dtheta]), pos=np.array([self.r, self.theta]),
                              current_time=current_time)


def print_trajectory_for_test(real):
    file_name = '2dtraj'
    if real:
        simulator = TwoDSimulator()
    else:
        simulator = FakeSimulator(file_name)
    positions = []
    satellite_states = []
    num_steps = int(3600 * 24* 10 / simulator.dt)
    for step in range(num_steps):
        satellite_state = simulator.simulate(step)
        if satellite_state is None:
            print("Simulation stopped: Satellite has crashed into the Earth.")
            break
        positions.append(utilities.p_to_c(satellite_state.pos))
        satellite_states.append(satellite_state)
    positions = np.array(positions)
    earth_radius = config['earth']['major_axis']  # Earth's radius

    # Plotting the trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], label='Satellite Orbit')
    earth = plt.Circle((0, 0), earth_radius, color='blue', fill=False, linestyle='dashed', linewidth=2, label='Earth')
    plt.gca().add_patch(earth)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Orbit Simulation with Earth')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    if real:
        with open(file_name, 'wb') as file:
         pickle.dump(satellite_states, file)

# print_trajectory_for_test(False)


