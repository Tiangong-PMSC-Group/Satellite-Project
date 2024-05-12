import numpy as np
import math 
import matplotlib.pyplot as plt
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
        self.grav_const = config['earth']['gravitational_constant']


    def ellipse_equation(self, x, y, z):
        """Defines the ellipse equation

        Args:
            x (float): Coordinate x
            y (float): Coordinate y
            z (float): Coordinate z

        Returns:
            float: _description_
        """
        return (x**2)/(self.re**2) + (y**2)/(self.re**2) + (z**2)/(self.rp**2) - 1


    def distance_to_surface(self, state, tol = 1e-3, max_iter = 20):
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
        t = np.arctan2(x2[1], x2[0])

        ell = (x[0] ** 2)/(self.re ** 2) + (x[1] ** 2)/(self.rp ** 2) -1
        Inside = False
        if ell < 0: 
            Inside = True

            
        iterations = 0
        error = tol
                
        while (error >= tol) and (iterations < max_iter):
            cost = np.cos(t)
            sint = np.sin(t)

            x1 = np.array([self.re * cost, self.rp * sint])
            xp = np.array([-self.re * sint, self.rp * cost])
            xpp = np.array([-self.re * cost, -self.rp * sint])
            delta = x1 - x2
            dp = np.dot(xp, delta)
            dpp = np.dot(xpp, delta) + np.dot(xp, xp)

            t -= dp / dpp
            error = abs(dp / dpp)
            iterations += 1
        

        #Makes it return a point on the ellipse rather than the optimisation
        opt_x = self.re * np.cos(t)
        opt_y = self.rp * np.sin(t)
        distance = np.sqrt((x2[0] - opt_x)**2 + (x2[1] - opt_y)**2)

        return dict(distance = distance, x = opt_x, y = opt_y, inside = Inside, error = error)

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
    



    ### Bonus Question

    def create_city(center, density_param,):
        pass



class City():
    def __init__(self, center, density = None, cov_matrix = None, cut_off = None, precision = 1000):
        
        self.center = center
    
        if density is None:
            params = config['city']['density_params']
            self.dens = np.random.lognormal(params[0], params[1])
        else:
            self.dens = density

        if cov_matrix is None:
            params = config['city']['cov_params']
            print(params)
            cov = 1/np.random.gamma(params[1], params[2])
            print(cov)
            self.cov = params[0] * np.array([[1, cov], [cov, 1]])
            print(self.cov)
        else:
            self.cov = cov_matrix

        if cut_off is None:
            # cut_off in degrees
            params = config['city']['cut_off_params']
            self.cut = np.random.normal(params[0], params[1])
            print(self.cut)
        else:
            self.cut = cut_off


        self.grid = self.population_grid(precision)


    def population_density(self, target):
    
        # Our position array has the distance in the first from the (0,0,0) in meters.
        # We don't need that, only the angles (that translate to latitude and longitude)
        # This check allows to pass the usual array, or just the angles 
        if len(target) == 3:
            target = target[1:]
        if len(self.center) == 3:
            center = self.center[1:]
        else:
            center = self.center

        diff = target - center


        return self.dens * np.exp(-0.5 * diff.T @ np.linalg.inv(self.cov) @ diff)


    def population_grid(self, precision):

        x_values = np.linspace(self.center[1] - self.cut, self.center[1] + self.cut, precision)
        y_values = np.linspace(self.center[2] - self.cut, self.center[2] + self.cut, precision)
        Z = np.zeros((len(x_values), len(y_values)))

        for i in range(len(x_values)):
            for j in range(len(y_values)):
                point = [x_values[i], y_values[j]]
                Z[j, i] = self.population_density(point)

        return Z
    

    def get_population(self, target):

        x_min = self.center[1] - 2*self.cut
        y_min = self.center[2] - 2*self.cut
        x_max = self.center[1] + 2*self.cut
        y_max = self.center[2] + 2*self.cut

        x_step = (x_max - x_min) / (self.precision - 1)
        y_step = (y_max - y_min) / (self.precision - 1)

        x_index = int((target[1] - x_min) / x_step)
        y_index = int((target[2] - y_min) / y_step)

        return self.grid[y_index, x_index]


    def plot_heatmap(self):
        x_min = self.center[1] - 20*self.cut
        y_min = self.center[2] - 20*self.cut
        x_max = self.center[1] + 20*self.cut
        y_max = self.center[2] + 20*self.cut

        min_coord = min(x_min, y_min)
        max_coord = max(x_max, y_max)

        plt.imshow(self.grid , extent=[min_coord, max_coord, min_coord, max_coord], origin='lower', cmap='hot', aspect='auto')
        plt.colorbar()  # Show color scale
        plt.title('Heatmap of Population Density')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()