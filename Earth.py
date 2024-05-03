import numpy as np
import math 
import matplotlib as plt
from decorators import singleton
import utilities as ut


@singleton
class Earth():
    def __init__(self):
        
        '''
        Properties of the Earth and its position.
        '''

        self.re = 6378136.6
        self.rp = 6356751.9
        self.omega = 7.2921159 * 10**(-5)  
        self.mass = 5.9722*10**24


    def ellipse_equation(self, x, y, z):
        """Defines the ellipse equation

        Args:
            x (float): Coordinate x
            y (float): Coordinate y
            z (float): Coordinate z

        Returns:
            float: _description_
        """
        return (x^2)/(self.re^2) + (y^2)/(self.rp^2) + (z^2)/(self.re^2) - 1 

    def distane_to_surface(self, state, tolerance = 1e-6, max_iterations = 1000):
        """Finds the minimum distance between the specified point and the ellipse
        using Newton's method.

        NEEDS TO BE OPTIMISED FOR POLAR COORDINATES
        '''

        '''
        x comes in as a set of 3D polar coordinates, since we want it to be in cartesian 

        x1 - ellipse coordinate
        x2 - satellite coordinates

        Args:
            state (_type_): _description_
            tolerance (_type_, optional): _description_. Defaults to 1e-6.
            max_iterations (int, optional): _description_. Defaults to 1000.

        Returns:
            float: minimun distance to Earth
        """

        '''
        
        ''' 

        #We translate into the plane which is relevant to us
        x = np.asarray(state)
        relevant_state = np.array([x[0], x[1]])

        x2 = ut.p_to_c(relevant_state)

        t = math.atan2(x2[1], x2[0])

        a = self.re
        b = self.rp
            
        iterations = 0
        error = tolerance
        errors = []
        ts = []
                
        while error >= tolerance and iterations < max_iterations:
            cost = math.cos(t)
            sint = math.sin(t)
            x1 = np.array([a * cost, b * sint])
            xp = np.array([-a * sint, b * cost])
            xpp = np.array([-a * cost, -b * sint])
            delta = x1 - x2
            dp = np.dot(xp, delta)
            dpp = np.dot(xpp, delta) + np.dot(xp, xp)

            t -= dp / dpp
            error = abs(dp / dpp)
            errors.append(error)
            ts.append(t)
            iterations += 1
        
        ts = np.array(ts)
        errors = np.array(errors)
        y = np.linalg.norm(x1 - x2)

        #Makes it return a point on the ellipse rather than the optimisation
        opt_x = a * np.cos(t)
        opt_y = b * np.sin(t)
        distance = np.sqrt((x2[0] - opt_x)**2 + (x2[1] - opt_y)**2)

        success = error < tolerance and iterations < max_iterations
        return dict(distance = distance, x = opt_x, y = opt_y, error = error, iterations = iterations, success = success, xs = ts,  errors = errors)

    def air_denisty(self, distance):
        """_summary_

        Args:
            distance (_type_): _description_

        Returns:
            _type_: _description_
        """
        '''
        Atmosphere code, gets you the density.
        '''
        R = 8.3144598
        G = 9.80665
        M = 0.028964425278793993
        rl = {'Pb': [1.2250, 3.6392e-1, 1.9367e-1, 1.2165e-1, 7.4874e-2, 3.9466e-2, 0.01322, 3.8510e-3, 0.00143, 4.7526e-4, 0.00086, 2.8832e-4, 1.4934e-4, 0.000064, 2.3569e-5, 1.0387e-5, 4.3985e-6, 1.8119e-6, 7.4973e-7, 3.1593e-7, 1.4288e-7, 9.708e-8, 4.289e-8, 2.222e-8, 1.291e-8, 8.152e-9, 5.465e-9, 3.831e-9, 2.781e-9, 2.076e-9, 1.585e-9, 1.233e-9, 9.750e-10, 7.815e-30, 6.339e-10], 
            'Tb': [288.15, 216.650, 216.650, 216.650,217.650, 221.650, 228.65, 251.050, 270.65, 256.650, 270.65, 245.450, 231.450, 214.65, 208.399, 198.639, 188.893, 186.87, 188.42, 195.08, 208.84, 240.00, 300, 360, 417.23, 469.27, 516.59, 559.63, 598.78, 634.39, 666.80, 696.29, 723.13, 747.57, 769.81], 
            'hb': [0, 11e3, 15e3, 18e3, 21e3, 25e3, 32e3, 40e3, 47e3, 51e3, 56e3, 60e3, 65e3, 71e3, 75e3, 80e3, 85e3, 90e3, 95e3, 100e3, 105e3, 110e3, 115e3, 120e3, 125e3, 130e3, 135e3, 140e3, 145e3, 150e3, 155e3, 160e3, 165e3, 170e3, 175e3]}

        layer = 0
        if distance > rl['hb'][-1]:
             rho = 6e-10*np.exp(-(distance - 175e3)/29.5e3)    
        else:        
            for i in range(len(rl['hb']) - 1):
                if distance >= rl['hb'][i] and distance < rl['hb'][i + 1]:
                    layer = i
            scale_height = R*rl['Tb'][layer]/(M*G)
            rho = rl['Pb'][layer]*np.exp(-(distance - rl['hb'][layer])/scale_height)
        return rho
