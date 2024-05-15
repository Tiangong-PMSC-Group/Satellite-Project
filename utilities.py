import numpy as np
import scipy as sp
import math
import random
from config import config



##### Converters #####

def c_to_p(state):
    """Changes the coordinates from Cartesian to 2D Polar or 3D Spherical

    Args:
        state (numpy.array): Cartesian coordinates array 

    Returns:
        numpy.array:  2D Polar (rho, polar) or 3D Spherical (rho, polar, azimuthal) array
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
            assert True, "Point lies on the the z-axis, Azimuthal angle cannot be defined, try a new inclination!"
    
        if x == y == z == 0:
            rho = 0
            polar = 0
            azimuthal = 0
        
        assert math.isinf(x) == False, "Unstable Orbit: a function is receiving an infinty instead of number, try some different values!"
        assert math.isinf(y) == False, "Unstable Orbit: a function is receiving an infinty instead of number, try some different values!"
        assert math.isinf(z) == False, "Unstable Orbit: a function is receiving an infinty instead of number, try some different values!"

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
        else:
            azimuthal = 0


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
        state (numpy.array): 2D Polar (rho, polar) or 3D Spherical (rho, polar, azimuthal) array

    Returns:
        numpy.array: Cartesian coordinates array
    """

    dims = np.size(state)
    
    if dims == 3:
        rho = state[0]
        polar = state[1]
        azimuthal = state[2]

        assert math.isinf(rho) == False, "Unstable Orbit: a function is receiving an infinte radius of orbit, try some different values!"
        assert math.isinf(polar) == False, "Unstable Orbit: a function is receiving an undefined polar angle, try some different values!"
        assert math.isinf(azimuthal) == False, "Unstable Orbit: a function is receiving an undefined azimuthal angle, try some different values!"

        x = rho * np.sin(polar) * np.cos(azimuthal)
        y = rho * np.sin(polar) * np.sin(azimuthal)
        z = rho * np.cos(polar)

        cartesian_state = np.array([x, y, z])

    elif dims == 2:
        rho = state[0]
        theta = state[1]

        assert math.isinf(rho) == False, "Unstable Orbit: a function is receiving an infinte radius of orbit, try some different values!"
        assert math.isinf(theta) == False, "Unstable Orbit: a function is receiving an undefined polar angle, try some different values!"

        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        cartesian_state = np.array([x, y])

    return cartesian_state

def earth_to_orbit(states):
    '''Convrets from Earth polar coordinates into satellite polar coordinates

    Args:
        states (numpy.array([rho, polar, aziumthal1]): Satellite coordinates array in reference spherical earth axis

    Returns:
        numpy.array([rho, polar]): Satellite polar array, defined in known orbital inclination angle
    '''
    x, z, y = p_to_c(states)
    
    if x == y == z == 0:
        rho = 0
        polar = 0
    
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
        state (numpy.array([numpy.array([rho1, polar1, aziumthal1]), ...])): Satellite coordinates array in reference spherical obrit axis

    Returns:
        numpy.array([numpy.array([x1, y1, z1]), ...]): Satellite coordinates array in refernce Earths XYZ
    """   
    bulk_array = np.zeros_like(states)
    for i in range(len(bulk_array)):
        x, z, y = p_to_c(states[i])
        bulk_array[i] = np.array([x, y, z])

    return bulk_array

def earth_to_xyz_bulk(states):
    """Changes the coordinates from the Earths spherical axis to Earths XYZ

    Args:
        state (numpy.array([numpy.array([rho1, polar1, aziumthal1]), ...])): Earth coordinates array in reference spherical Earth axis

    Returns:
        numpy.array([numpy.array([x1, y1, z1]), ...]): Earth coordinates array in refernce Earths XYZ
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
        state (numpy.array([R, Polar, Azimuthal])): Sperical coordinates array in one axis (Earth or Satellite)

    Returns:
        numpy.array([R, Polar, Azimuthal]): Sperical coordinates array in one axis (Earth or Satellite)
    """
    x, z, y = p_to_c(state)     #convert to cartesian, while switching the y and z coordinates
    rho, polar, azimuthal = c_to_p(np.array([x, y, z]))     #convert back to Sperical coordinates
    
    return np.array([rho, polar, azimuthal])

def bulk():
    pass


####################################################


def random_points_on_ellipse(earth, num_points):
    """Adds a number of randomly distributes points on Earths surface

    Args:
        earth (Earth Object): a class object Earth
        num_points (int): number of points

    Returns:
        numpy.array: array of points with 3D polar positions
    """ 
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
    """Calculates arc value between two points.

    Args:
        point1 (numpy.array): a point on Earth
        point2 (numpy.array): a ponit on Earth

    Returns:
        float: closest distance through ellipsoid
    """ 

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