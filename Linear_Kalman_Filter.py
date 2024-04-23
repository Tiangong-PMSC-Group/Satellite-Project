import numpy as np

class Linear_Kalman_Filter():
    def __init__(self, m0, Cov0, F, H, R = 1, Q = 0):
        '''
        m0 - inital guess of the means (Column Vector n x 1)
        Cov0 - intial covariance matrix, how good is the first guess (Square Matrix n x n)

        F - state transition matrix, the physics per timestep (Square Matrix n x n)
        R - covariance of observation noise (n x n)
        H - observation matrix (Binary Column Vector n x 1)
        Q - covariance of process noise (n x n)

        y - data recieved (Column Vector n x 1), look at update_mean(self, y)

        
        ### ### ### Operational Loop: ### ### ###

        1. Intialise/Reset the Filter

        2. If you DON'T recieve data (y):
            2.a. forecast()

        3. If you received data (y):
            3.a. update()

        4. Carry out 2 and 3 until exit condition is met.
        
        P.S. m and C are public variables so you can always access them through a = LKM.C and LKM.m
        '''

        #These don't really matter, make them large for large uncertainty
        self.m = m0
        self.C = Cov0

        #These govern the filter dynamics
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q 

    def forecast_mean(self):
        fore_mean = (self.F) @ self.m
        return fore_mean
    
    def forecast_cov(self):
        fore_cov = self.F @ (self.C) @ self.F.T + self.Q
        return fore_cov
    
    #The place where you feed the new recieved data in; y (Column Vector n x 1)
    def update_mean(self, y):
        K = self.get_KalmanGain()
        update_mean = self.m + K @ (y - self.H @ self.m)
        return update_mean
    
    def update_cov(self):
        K = self.get_Kalman_Gain()
        update_cov = self.C - K @ self.H @ self.C
        return update_cov

    def get_Kalman_Gain(self):
        #CHANGE THIS LINE TO SOLVE INSTEAD OF LINALG.INV
        K = self.C @ self.H.T @ np.linalg.inv(self.H @ self.C @ self.H.T + self.R)
        #K = self.C @ self.H.T @ np.linalg.solve(self.H @ self.C @ self.H.T + self.R, self.R)
        return K
    

    def forecast(self):
        self.m = self.forecast_mean()
        self.C = self.forecast_cov()

    def update(self, y):
        self.m = self.update_mean(y=y)
        self.C = self.update_cov()

    def reset_Filter(self, m0, Cov0):
        self.m = m0
        self.C = Cov0