import matplotlib
matplotlib.use('TkAgg')
from RadarSystem import RadarSystem
import utilities
from Earth import Earth
from IPython.core.display_functions import clear_output
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, FloatSlider, Output, Button
import time

from config import config


class Visualisation:
    def __init__(self, x_positions, y_positions, z_positions, radar_positions=None):
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.z_positions = z_positions
        self.radarSystem = RadarSystem(500, Earth())


    re = Earth().re
    rp = Earth().rp
    show_range = 8000000

    def update_view(self, elev=30, azim=30):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-self.show_range * 1.1, self.show_range * 1.1])
        ax.set_ylim([-self.show_range * 1.1, self.show_range * 1.1])
        ax.set_zlim([-self.show_range * 0.1, self.show_range * 0.1])
        ax.set_xlim([-self.show_range, self.show_range])
        ax.set_ylim([-self.show_range, self.show_range])
        ax.set_zlim([-self.show_range, self.show_range])

        # Earth surface
        u = np.linspace(0, 2 * np.pi, 200)
        v = np.linspace(0, np.pi, 200)
        x = self.re * np.outer(np.cos(u), np.sin(v))
        y = self.re * np.outer(np.sin(u), np.sin(v))
        z = self.rp * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color='b', rstride=4, cstride=4, alpha=0.1)
        ax.plot(self.x_positions, self.y_positions, self.z_positions, c='g')



        view_xs = []
        view_ys = []
        view_zs = []

        for i in range(len(self.x_positions)):
            pos = np.array([self.x_positions[i],self.y_positions[i],self.z_positions[i]])
            pos1 = utilities.c_to_p(pos)
            # print(pos)
            # print(pos1)
            # print("pos1")
            states = self.radarSystem.try_detect_satellite(pos1, i)
            print(utilities.p_to_c(self.radarSystem.radars[0].position))
            if len(states) > 0:
                # print(states[0])
                c_pos = utilities.p_to_c(states[0].pos)
                view_xs.append(c_pos[0])
                view_ys.append(c_pos[1])
                view_zs.append(c_pos[2])

        ax.scatter(view_xs,view_ys,view_zs, color='r')
        # print("a")
        # print(view_xs)
        # print(view_ys)
        # print(view_zs)
        ax.view_init(elev=elev, azim=azim)
        # Plot radar positions
        for radar in self.radarSystem.radars:
            position = utilities.p_to_c(radar.position)
            ax.scatter(position[0], position[1], position[2], color='g', marker='o')
        plt.show()

print(config['radar']['noise']['rho'])
orbit_radius = Earth().re + 100000
theta = np.linspace(0, 2 * np.pi, 100)
orbit_x = orbit_radius * np.cos(theta)
orbit_y = orbit_radius * np.sin(theta)
orbit_z = np.zeros_like(orbit_x)
# Update the instantiation of the Visualisation class to include radar positions
visual = Visualisation(orbit_x, orbit_y, orbit_z)
visual.update_view()




