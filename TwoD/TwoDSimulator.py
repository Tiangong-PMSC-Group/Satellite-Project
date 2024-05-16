import numpy as np
import matplotlib.pyplot as plt
from config import config
import utilities
from SatelliteState import SatelliteState

'''
Deprecated 2D model, which was no longer maintained and used after the introduction of the 3D model.
This model assumes a circular orbit around the equator and considers a simple atmospheric drag model.
'''
class TwoDSimulator:
    def __init__(self):
        """Initialize the 2D simulator with initial conditions and constants."""
        super().__init__()
        # Initial distance of the satellite from Earth's center
        self.r = config['satellite']['initial_conditions']['distance']
        self.G = config['earth']['gravitational_constant']  # Gravitational constant
        self.M = config['earth']['mass']  # Mass of the Earth
        self.dt = config['sim_config']['dt']['main_dt']  # Time step for simulation
        self.A = config['satellite']['area'] # Cross-sectional area of the satellite
        self.Cd = config['satellite']['drag_coefficient']  # Drag coefficient for the satellite
        # Initial angular velocity which makes the satellite orbit Earth in a circular path
        self.dtheta = np.sqrt(self.G * self.M / self.r ** 3)
        self.theta = 0  # Initial angle
        self.dr = 0  # Initial radial velocity
        self.earth_radius = config['earth']['major_axis']

    def simulate(self,use_drag=False) -> SatelliteState:
        """Simulate one time step of the satellite's motion and return its state.

        Args:
            use_drag (bool): If True, include atmospheric drag in the simulation. Defaults to False.

        Returns:
            SatelliteState: The state of the satellite after one time step, including velocity and position.
        """
        if use_drag is True:
            # Calculate atmospheric density based on altitude
            if self.r > self.earth_radius:
                altitude = self.r - self.earth_radius
                rho = 1.225 * np.exp(-altitude / 8500)  # Density decreases exponentially with altitude
            else:
                rho = 0  # No atmosphere below Earth's surface

            # Calculate current speed
            v = np.sqrt(self.dr ** 2 + (self.r * self.dtheta) ** 2)
            # Calculate drag force
            Fd = -0.5 * self.Cd * rho * v ** 2 * self.A  # Drag force
            ad = Fd / self.M  # Acceleration due to drag
        else:
            ad = 0

        # Update dynamics with drag force
        new_dr = self.dr - self.dt * (self.r * self.dtheta ** 2 - self.G * self.M / self.r ** 2) + self.dt * ad
        new_dtheta = self.dtheta + self.dt * (2 * self.dr * self.dtheta / self.r)

        # Update position
        new_r = self.r + self.dt * new_dr
        new_theta = self.theta + self.dt * new_dtheta

        # Update state variables
        self.r, self.theta, self.dr, self.dtheta = new_r, new_theta, new_dr, new_dtheta

        # Return the new state of the satellite
        return SatelliteState(velocity=np.array([self.dr, self.dtheta]), pos=np.array([self.r, self.theta]))


def print_trajectory_for_test():
    """Simulate the satellite trajectory and plot it.

    This function initializes the 2D simulator, simulates the satellite's trajectory over 24 hours,
    and plots the resulting orbit along with the Earth.
    """
    simulator = TwoDSimulator()
    positions = []
    satellite_states = []
    num_steps = int(3600 * 24 / simulator.dt)  # Number of simulation steps for 24 hours

    for step in range(num_steps):
        satellite_state = simulator.simulate()
        positions.append(utilities.p_to_c(satellite_state.pos))  # Convert polar to Cartesian coordinates
        satellite_states.append(satellite_state)

    positions = np.array(positions)

    # Plotting the satellite's trajectory
    earth_radius = config['earth']['major_axis']  # Earth's radius
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], label='Satellite Orbit')
    earth = plt.Circle((0, 0), earth_radius, color='blue', fill=False, linestyle='dashed', linewidth=2, label='Earth')
    plt.gca().add_patch(earth)  # Add Earth circle to the plot
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Orbit Simulation with Earth')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# Run the simulation and plot the trajectory
print_trajectory_for_test()
