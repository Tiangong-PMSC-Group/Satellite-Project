import numpy as np
import scipy as sp
import math
import random
from config import config


def c_to_p(state):
    """Changes the coordinates from Cartesian to Polar

    Args:
        state (np.array): Cartesian coordinates array 

    Returns:
        np.array: Polar coordinates array
    """

    dims = np.size(state)

    if dims == 3:
        x = state[0]
        y = state[1]
        z = state[2]
        rho = np.sqrt(x**2 + y**2 + z**2)  # Radial distance
        theta = np.arccos(z / rho)          # Inclination angle (polar angle) from the z-axis
        phi = np.arctan2(y, x)              # Azimuthal angle from the x-axis
        polar_state = np.array([rho, theta, phi])

    elif dims == 2:
        
        x = state[0]
        y = state[1]
        rho = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x) 
        polar_state = np.array([rho, theta])

    return polar_state



def p_to_c(state):
    """Changes the coordinates from Polar to Cartesian

    Args:
        state (np.array): Polar coordinates array

    Returns:
        np.array: Cartesian coordinates array
    """

    dims = np.size(state)

    if dims == 3:
        rho = state[0]
        theta = state[1]
        phi = state[2]
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        cartesian_state = np.array([x, y, z])

    elif dims == 2:
        rho = state[0]
        theta = state[1]
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        cartesian_state = np.array([x, y])

    return cartesian_state


def random_points_on_ellipse(earth, num_points):
    points = []
    for _ in range(num_points):
        # Squared first eccentricity
        e2 = 1 - (earth.rp ** 2 / earth.re ** 2)

        # Generate random latitude and longitude
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)

        # Convert to radians
        phi = np.radians(latitude)
        lambda_ = np.radians(longitude)

        # Calculate the prime vertical radius
        N = earth.re / np.sqrt(1 - e2 * np.sin(phi) ** 2)

        # Convert to geocentric Cartesian coordinates
        x = N * np.cos(phi) * np.cos(lambda_)
        y = N * np.cos(phi) * np.sin(lambda_)
        z = (N * (1 - e2)) * np.sin(phi)

        # Append the converted polar coordinates of the point
        points.append(c_to_p(np.array([x, y, z])))
    return points



def distance_on_surface(point1, point2):
    # Checked against https://www.onlineconversion.com/map_greatcircle_distance.htm
    # Minimum difference probabily due to different earth major axis.
    # Receive points in polar coordinates, remove the radial distance
    p1 = point1[1:]
    p2 = point2[1:]

    mean_coord = (p1-p2)/2

    sin2 = np.sin(mean_coord)**2
    sqrt = np.sqrt(sin2[0]+np.cos(p1[0])*np.cos(p2[0])*sin2[1])
    d = 2* config['earth']['major_axis'] * np.arcsin(sqrt)
    return d

def population_density(target, center, density_param, cov_matrix):
    if len(target) == 3:
        target = target[1:]
    if len(center) == 3:
        center = center[1:]

    diff = target - center
    return density_param * np.exp(-0.5 * diff.T @ np.linalg.inv(cov_matrix) @ diff)





