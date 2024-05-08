import numpy as np

class LinearKalmanFilter():
    def __init__(self, mean_0, cov_0, transition_matrix, observation_matrix, observation_noise = 1, process_noise = 0):
        """_summary_

        Args:
            mean_0 (_type_): _description_
            cov_0 (_type_): _description_
            transition_matrix (_type_): _description_
            observation_matrix (_type_): _description_
            observation_noise (int, optional): _description_. Defaults to 1.
            process_noise (int, optional): _description_. Defaults to 0.
        """

        
        '''
        n_stat = number of total states
        n_mes = number of measured states

        m0 - inital guess of the means (Column Vector n_stat x 1)
        Cov0 - intial covariance matrix, how good is the first guess (Square Matrix n_stat x n_stat)

        F - state transition matrix, the physics per timestep (Square Matrix n_stat x n_stat)
        H - observation matrix (Binary Column Vector n_mes x n_stat)
        R - covariance of observation noise (n_mes x n_mes) noise of RADAR
        Q - covariance of process noise (n_stat x n_stat) noise of ATMOSPHERE/PHYSICS 

        measurement - data recieved (Column Vector n x 1), look at update_mean(self, y)


        ### ### ### Operational Loop: ### ### ###

        1. Intialise/Reset the Filter

        2. If you DON'T recieve data (y):
            2.a. LKM_forecast()

        3. If you received data (y):
            3.a. LKM_update()

        4. Carry out 2 and 3 until exit condition is met.
        
        P.S. m and C are public variables so you can always access them through a = LKM.C and LKM.m
        '''

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
        if transition_matrix is None:
            transition_matrix = self.F
        return (transition_matrix) @ self.m
        
    def forecast_cov(self, transition_matrix = None, process_noise = None):
        if transition_matrix is None:
            transition_matrix = self.F
        if process_noise is None:
            process_noise = self.Q  
        fore_cov = transition_matrix @ (self.C) @ transition_matrix.T + process_noise
        return fore_cov
    
    #The place where you feed the new recieved data in; y (Column Vector n x 1)
    def update_mean(self, measurement):
        K = self.get_kalman_gain()
        return self.m + K @ (measurement - self.H @ self.m)
        
    def update_cov(self):
        K = self.get_kalman_gain()
        return self.C - K @ self.H @ self.C
        
    def get_kalman_gain(self):
        #CHANGE THIS LINE TO SOLVE INSTEAD OF LINALG.INV
        return self.C @ self.H.T @ np.linalg.inv(self.H @ self.C @ self.H.T + self.R)
        #K = self.C @ self.H.T @ np.linalg.solve(self.H @ self.C @ self.H.T + self.R, self.R)

    # Project without new information
    def forecast(self, transition_matrix = None):
        self.m = self.forecast_mean(transition_matrix)
        self.C = self.forecast_cov(transition_matrix)
        return self.m, self.C

    # Receive the information
    def update(self, measurement):
        if measurement.ndim == 1:
            measurement = measurement[:, np.newaxis]

        self.m = self.update_mean(measurement)
        self.C = self.update_cov()
        return self.m, self.C

    def reset(self, m0, Cov0):
        self.m = m0
        self.C = Cov0


class ExtendedKalmanFilter(LinearKalmanFilter):
    def __init__(self, mean_0, cov_0, transition_matrix, observation_matrix = None, observation_noise = 1, process_noise = 0):
        
        if mean_0.ndim == 1:
            self.m = mean_0[:, np.newaxis]
        else:
            self.m = mean_0

        self.C = cov_0
        self.F = self.get_F(rho, dt)

        if observation_matrix == None:
            self.H = np.array([[1, 0, 0, 0, 0, 0,],
                               [0, 0, 0, 1, 0, 0,],])
        else:
            self.H = self.H()

        self.R = observation_noise
        self.Q = process_noise
        
        '''TODO:
            1. -Add these as properties of other classes
            2. - H matrix COULD (NOT SHOULD OR MUST, COULD) have observational functions to enchance the filter
        '''
        self.As = 1 
        self.Cd = 1
        self.ms = 1

        self.G = 1 
        self.Me = 1 


    def get_H(self):
        H = np.zeros([2, 6])
        return H
    
    def get_F(self, rho, dt):
        F = np.zeros([6, 6])

        '''TODO:
            1. - Import Earth and System Properties into the EKF to allow rho and dt extraction
        '''

        # FIRST ROW #
        F[0,0] = 1
        F[0,1] = dt

        # SECOND ROW #
        F[1,1] = 1 
        F[1,2] = dt

        # THIRD ROW #
        F[2,0] = (2 * self.G * self.Me * 1/(self.m[0] ** 3)) + (self.m[4]) ** 2
        F[2,5] = 2 * self.m[0] * self.m[4]

        # FORTH ROW #
        F[3,3] = 1
        F[3,4] = dt

        # FIFTH ROW #
        F[4, 4] = 1
        F[4, 5] = dt

        # SIXTH ROW #
        F[5,0] = self.As * self.Cd/self.ms *  -0.5 * rho * self.m[4] ** 2 
        F[5,4] = -rho * self.m[0] * self.m[4] * self.As * self.Cd/self.ms

        return F
    
    def forecast_mean(self, rho, dt):

        '''TODO:
            1. - Import Earth and System Properties into the EKF to allow rho and dt extraction
            2. - Add Runge Kutta to this (CURRENTLY: FORWARD EULER)
        '''

        self.m[0] = self.m[0] + dt * self.m[1]
        self.m[1] = self.m[1] + dt * self.m[2]
        self.m[2] = -self.G  * self.Me * 1/(self.m[0] ** 2) + self.m[0] * self.m[4] ** 2

    
        self.m[3] = self.m[3] + dt * self.m[4]
        self.m[4] = self.m[4] + dt * self.m[5]
        self.m[5] = -0.5 * rho * self.m[0] * (self.m[4] ** 2) * self.As * self.Cd/self.ms

    def forecast(self, rho, dt):

        F = self.get_F(rho, dt)
        self.m = self.forecast_mean(rho, dt)
        self.C = self.forecast_cov(transition_matrix=F, process_noise=None)

        return self.m, self.C
    