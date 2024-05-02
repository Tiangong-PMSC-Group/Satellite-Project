import numpy as np
import scipy as sp
import math

def c_to_p(state):
    '''
    Changes the coordinates from CARTESIAN to POLAR
    '''

    dims = np.size(state)

    if dims == 3:
        x = state[0]
        y = state[1]
        z = state[2]
        rho = np.sqrt(x**2 + y**2 + z**2)  # Radial distance
        theta = np.arccos(z / rho)          # Inclination angle (polar angle) from the z-axis
        phi = np.arctan2(y, x)              # Azimuthal angle from the x-axis
        polar_state = np.array([[rho], [theta], [phi]])

    elif dims == 2:
        x = state[0]
        y = state[1]
        rho = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x) 
        polar_state = np.array([[rho], [theta]])

    return polar_state
    
def p_to_c(state):
    '''
    Changes the coordinates from POLAR to CARTESIAN
    '''

    dims = np.size(state)

    if dims == 3:
        rho = state[0]
        theta = state[1]
        phi = state[2]
        x = rho * math.sin(phi) * math.cos(theta)
        y = rho * math.sin(phi) * math.sin(theta)
        z = rho * math.cos(phi)
        cartesian_state = np.array([[x], [y], [z]])

    elif dims == 2:
        rho = state[0]
        theta = state[1]
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        cartesian_state = np.array([[x], [y]])

    return cartesian_state