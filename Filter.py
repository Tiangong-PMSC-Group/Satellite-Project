import numpy as np
from Earth import Earth
from config import config
import utilities as ut


class LinearKalmanFilter():
    def __init__(self, mean_0, cov_0, transition_matrix, observation_matrix, observation_noise = np.eye(2), process_noise = np.eye(6)):
        """Generates the Linear Kalman Filter class.

        Args:
            mean_0 (numpy.array): initial guess of state 
            cov_0 (numpy.array): initial guess of state covariances
            transition_matrix (numpy.array): physics informed discretized dynamics
            observation_matrix (numpy.array): matrix of measurements coming in
            observation_noise (numpy.array, optional): noise of observations. Defaults to np.eye(2).
            process_noise (numpy.arrayt, optional): noise of the Kalman process. Defaults to np.eye(6).
        """

        if mean_0.ndim == 1:
            self.m = mean_0[:, np.newaxis]
        else:
            self.m = mean_0

        self.C = cov_0

        self.F = transition_matrix
        self.H = observation_matrix
        self.R = observation_noise
        self.Q = process_noise


    def forecast_mean(self, transition_matrix = None):
        """Forecasts the LKF state.

        Args:
            transition_matrix (numpy.array): transition matrix of the system. Defaults to None

        Returns:
            numpy.array: forecast mean by 1 dt
        """
        if transition_matrix is None:
            transition_matrix = self.F
        return (transition_matrix) @ self.m
        
    def forecast_cov(self, transition_matrix = None, process_noise = None):
        """Forecasts the LKF covariance matrix.

        Args:
            transition_matrix (numpy.array): transition matrix of the system. Defaults to None
            process_noise (numpy.array): matrix of the Kalman process noise. Defaults to None

        Returns:
            numpy.array: forecast covariance by 1 dt
        """
        if transition_matrix is None:
            transition_matrix = self.F
        if process_noise is None:
            process_noise = self.Q  
        fore_cov = transition_matrix @ (self.C) @ transition_matrix.T + process_noise
        return fore_cov
    
    #The place where you feed the new recieved data in; y (Column Vector n x 1)
    def update_mean(self, measurement):
        """Updates the LKF state.

        Args:
            measurement (numpy.array): measurement which corresponds to the observation matrix

        Returns:
            numpy.array: updated LKF state using the measurement
        """
        K = self.get_kalman_gain()
        #print(K)
        #print(measurement)
        #print(self.H)
        #print(self.m[0])
        #print('##############')
        return self.m + K @ (measurement - self.H @ self.m)
        
    def update_cov(self):
        """Updates the LKF covariance matrix.

        Returns:
            numpy.array: updated LKF covariance matrix using the measurement
        """
        K = self.get_kalman_gain()
        return self.C - K @ self.H @ self.C
        
    def get_kalman_gain(self):
        """Calcualtes the Kalman gain.

        Returns:
            numpy.array: Kalman gain
        """
        #CHANGE THIS LINE TO SOLVE INSTEAD OF LINALG.INV
        return self.C @ self.H.T @ np.linalg.inv(self.H @ self.C @ self.H.T + self.R)
        #K = self.C @ self.H.T @ np.linalg.solve(self.H @ self.C @ self.H.T + self.R, self.R)

    # Project without new information
    def forecast(self, transition_matrix = None):
        """Forecasts the LKF state and covariance matrix.

        Args:
            transition_matrix (numpy.array): transition matrix of the system. Defaults to None

        Returns:
            numpy.array: forecast mean by 1 dt
            numpy.array: forecast covariance by 1 dt
        """

        self.m = self.forecast_mean(transition_matrix)
        self.C = self.forecast_cov(transition_matrix)
        return self.m, self.C

    # Receive the information
    def update(self, measurement):
        """Updates the LKF covariance matrix and state.

        Args:
            measurement (numpy.array): measurement which corresponds to the observation matrix

        Returns:
            numpy.array: updated LKF covariance matrix using the measurement
            numpy.array: updated LKF state using the measurement
        """
        if measurement.ndim == 1:
            measurement = measurement[:, np.newaxis]
        
        self.m = self.update_mean(measurement)
        self.C = self.update_cov()
        return self.m, self.C

    def reset(self, m0, Cov0):
        """Resets the LKF state and covariance matrix

        Args:
            m0 (numpy.array): new LKF state
            Cov0 (numpy.array): new LKF covariance matrix
        """
        self.m = m0
        self.C = Cov0


class ExtendedKalmanFilter(LinearKalmanFilter):
    
    def __init__(self, mean_0, cov_0, planet, observation_matrix = None, observation_noise = np.eye(2), process_noise = np.eye(6)):
        """Generates the Extended Kalman Filter class.

        Args:
            mean_0 (numpy.array): initial guess of state 
            cov_0 (numpy.array): initial guess of state covariances
            planet (Earth Object): object of the planet currently simulated
            transition_matrix (numpy.array): physics informed discretized dynamics
            observation_matrix (numpy.array): matrix of measurements coming in. Defaults to None.
            observation_noise (numpy.array, optional): noise of observations. Defaults to np.eye(2).
            process_noise (numpy.arrayt, optional): noise of the Kalman process. Defaults to np.eye(6).
        """

        self.R = observation_noise
        self.Q = process_noise
        self.C = cov_0

        if mean_0.ndim == 1:
            self.m = mean_0[:, np.newaxis]
        else:
            self.m = mean_0

        if observation_matrix is None:
            self.H = np.array([[1, 0, 0, 0, 0, 0,],
                               [0, 0, 0, 1, 0, 0,],])
        else:
            self.H = observation_matrix

        self.planet = planet

        self.orbital_angle = np.pi/2 - config['satellite']['initial_conditions']['polar_angle']
        self.dt = config['sim_config']['dt']['main_dt']
        self.As = config["satellite"]["area"]
        self.Cd = config["satellite"]["drag_coefficient"]
        self.ms = config["satellite"]["mass"]
        self.G = config['earth']['gravitational_constant']
        self.Me = config['earth']['mass']

        self.F, rho = self.get_F()
    
    def get_F(self):
        """Calculates the transition matrix and density.

        Returns:
            numpy.array: discretised transition matrix for the ith step of EKF
            float: atmospheric density according to the EKF state
        """
        sat_coords = np.array([self.m[0][0], self.m[3][0], self.orbital_angle])
        earth_coords = ut.spherical_to_spherical(sat_coords)
        res = self.planet.distance_to_surface(state=earth_coords)

        rho = self.planet.air_density(res['distance'])

        F = np.zeros([6, 6])

        # FIRST ROW #
        F[0,0] = 1
        F[0,1] = self.dt

        # SECOND ROW #
        F[1,1] = 1 
        F[1,2] = self.dt

        # THIRD ROW #
        F[2,0] = (2 * self.G * self.Me * 1/(self.m[0] ** 3)) + (self.m[4]) ** 2
        F[2,4] = 2 * self.m[0] * self.m[4]

        # FORTH ROW #
        F[3,3] = 1
        F[3,4] = self.dt

        # FIFTH ROW #
        F[4, 4] = 1
        F[4, 5] = self.dt

        # SIXTH ROW #
        F[5,0] = self.As * self.Cd/self.ms *  -0.5 * rho * self.m[4] ** 2 
        F[5,4] = -rho * self.m[0] * self.m[4] * self.As * self.Cd/self.ms

        return F, rho

    def forecast_mean(self, rho):
        """Forecasts the EKF state.

        Args:
            rho (float): atmospheric density at ith step

        Returns:
            numpy.array: forecast EKF state by 1 dt 
        """

        '''TODO:
        2. - Add Runge Kutta to this (CURRENTLY: LEAPFROG EULER FORWARD)
        '''

        m = np.zeros(6)
        m[0] = self.m[0] + 0.5 * self.dt * self.m[1]
        m[1] = self.m[1] + 0.5 * self.dt * self.m[2]
        m[2] = -self.G  * self.Me * 1/(self.m[0] ** 2) + self.m[0] * self.m[4] ** 2

        m[3] = self.m[3] + 0.5 * self.dt * self.m[4]
        m[4] = self.m[4] + 0.5 * self.dt * self.m[5]
        m[5] = -0.5 * rho * self.m[0] * (self.m[4] ** 2) * self.As * self.Cd/self.ms

        m = m[:, np.newaxis]
        return m
    
    def forecast(self):
        """Forecasts the EKF state and covariance matrix.

        Returns:
            numpy.array: forecast EKF state by 1 dt 
            numpy.array: forecast EKF covariance by 1 dt
        """
        F, rho = self.get_F()
        self.m = self.forecast_mean(rho=rho)
        self.C = self.forecast_cov(transition_matrix=F, process_noise=None)

        return self.m, self.C
    