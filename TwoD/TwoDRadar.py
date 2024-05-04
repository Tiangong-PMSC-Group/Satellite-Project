import numpy as np
import math

import utilities
import utilities as ut
from Radar import Radar
from config import config

class TwoDRadar(Radar):
    def __init__(self, position):
        super().__init__(position)

    def earth_equation(self,x, y):
        earth_radius = 6371000  # meters
        return x ** 2 + y ** 2 <= earth_radius ** 2

    def line_of_sight(self, Sat_Pos, Earth_Eqn):
        '''
        Check line of sight in TwoD between radar and satellite.
        Sat_Pos: Satellite position in TwoD (x, y)
        Earth_Eqn: A function that checks if a point is within the Earth in TwoD.
        '''

        # Convert positions from polar to Cartesian if necessary (assuming it's already Cartesian here)
        x = utilities.p_to_c(Sat_Pos)
        p = utilities.p_to_c(self.position)

        # Calculate direction vector and magnitude
        v0 = np.array(x) - np.array(p)
        mag = np.linalg.norm(v0)
        v0 = v0 / mag

        # Check if the line intersects with Earth
        for t in np.linspace(0, mag, 10):
            LOS_p = np.array(p) + v0 * t
            if self.earth_equation(LOS_p[0], LOS_p[1]):
                return False

        return True

