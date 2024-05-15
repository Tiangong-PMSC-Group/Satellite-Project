import math
import numpy as np
from matplotlib import pyplot as plt
import utilities
from Earth import Earth
from The_Pit.ISimulator import ISimulator, FakeSimulator, RealSimulator
from RadarSystem import RadarSystem
from TwoD.TwoDRadarSystem import TwoDRadarSystem
from config import config
from TwoD.TwoDSimulator import TwoDSimulator
'''
Code just for test
'''
class Test2:
    def __init__(self):
        is2D = True
        real_simulator = True
        if is2D:
            self.simulator = TwoDSimulator()
            self.radarSystem = TwoDRadarSystem(1, Earth())
        else:
            self.radarSystem = RadarSystem(1, Earth())
            if real_simulator:
                self.simulator = FakeSimulator()
            else:
                self.simulator = RealSimulator()


    def fall(self):
        num_steps = int(5600 / self.simulator.dt)
        positions = []
        for current_time in range(1, num_steps):
            satellite_state = self.simulator.simulate(current_time)
            if current_time % config['sim_config']['dt']['radar_los'] == 0:
                self.radarSystem.update_radar_positions()
                satellite_pos = self.radarSystem.try_detect_satellite(satellite_state.pos, current_time)
                if satellite_pos is not None:
                    positions.append(utilities.p_to_c(satellite_pos))

        positions = np.array(positions)
        plt.figure(figsize=(8, 8))
        plt.plot(positions[:, 0], positions[:, 1])
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('Orbit Simulation using Two_D_Simulator')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


Test2().fall()
