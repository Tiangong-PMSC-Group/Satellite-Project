from main import Main
import matplotlib.pyplot as plt
import numpy as np
from config import config
import utilities
main = Main(200)
main.simulate()
main.predict()
R, rad, R2, rad2, R3, rad3 = main.output()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(rad, R-6300000, c='b')

plt.show()

R, rad, R2, rad2, R3, rad3 = main.output()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(rad2, R2-6300000, c='b')

plt.show()

trd = [np.pi/2 - config['satellite']['initial_conditions']['polar_angle']]*len(R)
predicted_states_earth_cord = np.array([utilities.spherical_to_spherical(np.array([R,rad,trd]).T[i]) for i in range(len(R))])
real_x, real_y, real_z = utilities.earth_to_xyz_bulk(predicted_states_earth_cord).T

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.scatter(real_x, real_y, real_z)

plt.show()