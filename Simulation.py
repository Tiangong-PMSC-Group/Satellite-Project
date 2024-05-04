import math

import numpy as np

from Earth import Earth
from ISimulator import ISimulator, FakeSimulator, RealSimulator
from RadarSystem import RadarSystem
from TwoD.TwoDRadarSystem import TwoDRadarSystem
from config import config
from TwoD.TwoDSimulator import TwoDSimulator


def fall():
    is2D = True
    simulator = None
    radarSystem = None
    real_simulator = True
    if is2D:
        simulator = TwoDSimulator()
        radarSystem = TwoDRadarSystem(1, Earth())
    else:
        radarSystem = RadarSystem(1, Earth())
        if real_simulator:
            simulator = FakeSimulator()
        else:
            simulator = RealSimulator()

    for current_time in range(1, 10000):
        satellite_state = simulator.simulate(current_time)
        if current_time % config['sim_config']['dt']['radar_los'] == 0:
            radarSystem.update_radar_positions()
            radarSystem.try_detect_satellite(satellite_state.pos, current_time)




fall()
