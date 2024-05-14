import numpy as np
import scipy as sp
import math
import random
from config import config



##### Converters #####

def c_to_p(state):
    """Changes the coordinates from Cartesian to 2D Polar or 3D Spherical

    Args:
        state (np.array): Cartesian coordinates array 

    Returns:
        np.array:  2D Polar (rho, polar) or 3D Spherical (rho, polar, azimuthal) array
    """

    #dimension of input array (3D or 2D)
    dims = np.size(state)

    if dims == 3:

        x = state[0]
        y = state[1]
        z = state[2]

        rho = np.sqrt(x**2 + y**2 + z**2)  # Radial distance

        #calculates the polar angle measured from the x-axis
        if x == y == 0:
            assert True, "Point lies on the the z-axis, Azimuthal angle cannot be defined"
    
        if x == y == z == 0:
            assert True, "Point is on origin, Polar and Azimuthal angle cannot be defined"
        
        if z > 0:    
            polar = np.arctan(np.sqrt(x**2 + y**2)/z)
        elif z < 0:
            polar = np.pi + np.arctan(np.sqrt(x**2 + y**2)/ z)
        elif z == 0:
            polar = 0.5*np.pi
        
        #calculates the azimuthal angle measured from the orthogonal 
        #projection of r onto the reference x-y plane is direction of x to y.
        #does np.arctan2(y,x) make the first 3 check redundant?
        if x > 0:
            azimuthal = np.arctan(y/x)
        elif x < 0 and y >= 0:
            azimuthal = np.arctan(y/x) + np.pi
        elif x < 0 and y < 0:
            azimuthal = np.arctan(y/x) - np.pi
        elif x == 0 and y > 0:
            azimuthal = 0.5*np.pi
        elif x == 0 and y < 0:
            azimuthal = -0.5*np.pi

        polar_state = np.array([rho, polar, azimuthal])

    elif dims == 2:
        
        x = state[0]
        y = state[1]

        rho = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x) 
        polar_state = np.array([rho, theta])

    return polar_state



def p_to_c(state): 
    """Changes the coordinates from 2D Polar or 3D Spherical to Cartesian

    Args:
        state (np.array): 2D Polar (rho, polar) or 3D Spherical (rho, polar, azimuthal) array

    Returns:
        np.array: Cartesian coordinates array
    """

    dims = np.size(state)

    if dims == 3:
        rho = state[0]
        polar = state[1]
        azimuthal = state[2]

        assert math.isinf(rho) == False, "TypeError: function receiving infinity instead of a number for Radius"
        assert math.isinf(polar) == False, "TypeError: function receiving infinity instead of a number for Polar Angle"
        assert math.isinf(azimuthal) == False, "TypeError: function receiving infinity instead of a number for Azimuthal Angle"

        x = rho * np.sin(polar) * np.cos(azimuthal)
        y = rho * np.sin(polar) * np.sin(azimuthal)
        z = rho * np.cos(polar)

        cartesian_state = np.array([x, y, z])

    elif dims == 2:
        rho = state[0]
        theta = state[1]

        assert math.isinf(rho) == False, "TypeError: function receiving infinity instead of a number for Radius"
        assert math.isinf(theta) == False, "TypeError: function receiving infinity instead of a number for Polar Angle"

        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        cartesian_state = np.array([x, y])

    return cartesian_state

def earth_to_orbit(states):
    '''ShEs BAck
    Args:
        state (np.array([rho, polar, aziumthal1]): Satellite coordinates array in reference spherical earth axis

    Returns:
        np.array([rho, polar]): Satellite polar array, defined in known orbital inclination angle
    '''
    x, z, y = p_to_c(states)
    
    if x == y == 0:
        assert True, "Point lies on the the z-axis, Azimuthal angle cannot be defined"
    
    if x == y == z == 0:
        assert True, "Point is on origin, Polar and Azimuthal angle cannot be defined"
    
    rho = np.sqrt(x**2 + y**2 + z**2)
    
    if z > 0 and x > 0:    
        polar = np.arctan(np.sqrt(x**2 + y**2)/z)
    elif z < 0 and x > 0:
        polar = np.pi + np.arctan(np.sqrt(x**2 + y**2)/ z)
    elif z < 0 and x < 0:
        polar = np.pi - np.arctan(np.sqrt(x**2 + y**2)/ z)
    elif z > 0 and x < 0:
        polar = 2*np.pi - np.arctan(np.sqrt(x**2 + y**2)/ z)
    elif z == 0  and x > 0:
        polar = 0.5*np.pi
    elif z == 0 and x < 0:
        polar = 1.5*np.pi
    elif z < 0 and x == 0:
        polar = np.pi
    elif z > 0 and x == 0:
        polar = 0
    
    return np.array([rho, polar])


def satellite_to_xyz_bulk(states):    
    """Changes the coordinates from the Orbits spherical axis to Earths XYZ

    Args:
        state (np.array([np.array([rho1, polar1, aziumthal1]), ...])): Satellite coordinates array in reference spherical obrit axis

    Returns:
        np.array([np.array([x1, y1, z1]), ...]): Satellite coordinates array in refernce Earths XYZ
    """   
    bulk_array = np.zeros_like(states)
    for i in range(len(bulk_array)):
        x, z, y = p_to_c(states[i])
        bulk_array[i] = np.array([x, y, z])

    return bulk_array

def earth_to_xyz_bulk(states):
    """Changes the coordinates from the Earths spherical axis to Earths XYZ

    Args:
        state (np.array([np.array([rho1, polar1, aziumthal1]), ...])): Earth coordinates array in reference spherical Earth axis

    Returns:
        np.array([np.array([x1, y1, z1]), ...]): Earth coordinates array in refernce Earths XYZ
    """ 
    bulk_array = np.zeros_like(states)
    for i in range(len(bulk_array)):
        x, y, z = p_to_c(states[i])
        bulk_array[i] = np.array([x, y, z])

    return bulk_array


#HEar me out orbit_to_earth2.0?? :3
def spherical_to_spherical(state):
    """Changes the coordinates from one Spherical axis to Spherical another axis

    Args:
        state (np.array([R, Polar, Azimuthal])): Sperical coordinates array in one axis (Earth or Satellite)

    Returns:
        np.array([R, Polar, Azimuthal]): Sperical coordinates array in one axis (Earth or Satellite)
    """
    x, z, y = p_to_c(state)     #convert to cartesian, while switching the y and z coordinates
    rho, polar, azimuthal = c_to_p(np.array([x, y, z]))     #convert back to Sperical coordinates
    
    return np.array([rho, polar, azimuthal])

def bulk():
    pass


####################################################


def random_points_on_ellipse(earth, num_points):
    points = []
    for _ in range(num_points):
        # Squared first eccentricity
        e2 = 1 - (earth.rp ** 2 / earth.re ** 2)

        # Generate random latitude and longitude
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)

        # Convert to radians
        polar = np.radians(latitude)
        azimuthal = np.radians(longitude)

        # Calculate the prime vertical radius
        N = earth.re / np.sqrt(1 - e2 * np.sin(polar) ** 2)

        # Convert to geocentric Cartesian coordinates
        x = N * np.cos(polar) * np.cos(azimuthal)
        y = N * np.cos(polar) * np.sin(azimuthal)
        z = (N * (1 - e2)) * np.sin(polar)

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