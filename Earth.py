import numpy as np
import math 
import matplotlib as plt
from decorators import singleton
import utilities as ut
from config import config


@singleton
class Earth():
    def __init__(self):
        
        '''
        Properties of the Earth and its position.
        '''

        self.re = config['earth']['major_axis']
        self.rp = config['earth']['minor_axis']
        self.omega = config['earth']['omega']
        self.mass = config['earth']['mass']


    def ellipse_equation(self, x, y, z):
        """Defines the ellipse equation

        Args:
            x (float): Coordinate x
            y (float): Coordinate y
            z (float): Coordinate z

        Returns:
            float: _description_
        """
        return (x**2)/(self.re**2) + (y**2)/(self.rp**2) + (z**2)/(self.re**2) - 1


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
                
        while (error >= tolerance) and (iterations < max_iterations):
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

    def air_density(self, distance):
        """_summary_

        Args:
            distance (_type_): _description_

        Returns:
            _type_: _description_
        """
        '''
        Atmosphere code, gets you the density.
        '''
        R = config['atmosphere']['gas_const']
        G = config['atmosphere']['gravity']
        M = config['atmosphere']['molar_mass']
        rl = config['atmosphere']['layers']
        layer = 0
        if distance > rl['hb'][-1]:
             rho = config['atmosphere']['rho_params']['rho0']*np.exp(-(distance - config['atmosphere']['rho_params']['rho1'])/config['atmosphere']['rho_params']['rho2'])    
        else:        
            for i in range(len(rl['hb']) - 1):
                if distance >= rl['hb'][i] and distance < rl['hb'][i + 1]:
                    layer = i
            scale_height = R*rl['Tb'][layer]/(M*G)
            rho = rl['Pb'][layer]*np.exp(-(distance - rl['hb'][layer])/scale_height)
        return rho
