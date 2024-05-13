import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from config import config
from Earth import Earth
from main import Main
import utilities

matplotlib.use('TkAgg')

duration_time =10
transition_time = 10
class VisualisationPlotly:
    def __init__(self, states1, states2):
        self.states1 = states1
        self.states2 = states2
        self.re = Earth().re
        self.rp = Earth().rp
        self.show_range = 8000000

    def sphere(self, texture):
        N_lat = int(texture.shape[0])
        N_lon = int(texture.shape[1])
        theta = np.linspace(0, 2 * np.pi, N_lat)
        phi = np.linspace(0, np.pi, N_lon)

        x0 = self.re * np.outer(np.cos(theta), np.sin(phi))
        y0 = self.re * np.outer(np.sin(theta), np.sin(phi))
        z0 = self.rp * np.outer(np.ones(N_lat), np.cos(phi))

        return x0, y0, z0

    def create_earth_surface(self):
        colorscale = [[0.0, 'rgb(30, 59, 117)'],
                      [0.1, 'rgb(46, 68, 21)'],
                      [0.2, 'rgb(74, 96, 28)'],
                      [0.3, 'rgb(115,141,90)'],
                      [0.4, 'rgb(122, 126, 75)'],
                      [0.6, 'rgb(122, 126, 75)'],
                      [0.7, 'rgb(141,115,96)'],
                      [0.8, 'rgb(223, 197, 170)'],
                      [0.9, 'rgb(237,214,183)'],
                      [1.0, 'rgb(255, 255, 255)']]

        texture = np.asarray(Image.open('earth.jpg').resize((200, 200))).T
        x, y, z = self.sphere(texture)
        return go.Surface(x=x, y=y, z=z, surfacecolor=texture, colorscale=colorscale, opacity=0.5)

    def create_earth_surface1(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.re * np.outer(np.cos(u), np.sin(v))
        y = self.re * np.outer(np.sin(u), np.sin(v))
        z = self.rp * np.outer(np.ones(np.size(u)), np.cos(v))
        return go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale='Blues', showscale=False)

    def highlight_last_points(self, states, color):
        if len(states) > 1:
            x, y, z = zip(*states[-1:])
            return go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(color=color, size=5, symbol='circle', opacity=0.5),
                name="Final Location"
            )
        return None

    def create_trajectory_frames(self, states1, states2, color1, color2):
        frames = []
        for progress in range(101):
            progress_index = int(len(states1) * progress / 100)
            x1, y1, z1 = zip(*states1[:progress_index]) if progress_index > 0 else ([], [], [])
            x2, y2, z2 = zip(*states2[:progress_index]) if progress_index > 0 else ([], [], [])

            frame_data = [
                go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', line=dict(color=color1, width=2), opacity=0.5),
                go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', line=dict(color=color2, width=2), opacity=0.5)
            ]

            if progress == 100:
                last_point1 = self.highlight_last_points(states1, color1)
                last_point2 = self.highlight_last_points(states2, color2)
                if last_point1:
                    frame_data.append(last_point1)
                if last_point2:
                    frame_data.append(last_point2)
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers'))
                frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers'))

            frames.append(go.Frame(data=frame_data, name=str(progress), traces=[1, 2, 3, 4]))

        return frames

    def visualise(self):
        fig = go.Figure()

        earth_surface = self.create_earth_surface()
        fig.add_trace(earth_surface)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=2), opacity=0.5))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=2), opacity=0.5))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers'))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers'))

        fig.frames = self.create_trajectory_frames(self.states1, self.states2, 'red', 'green')

        sliders = [{
            'pad': {"t": 30},
            'steps': [{'args': [[str(k)], {'frame': {'duration': duration_time, 'redraw': True},
                                           'mode': 'immediate', 'transition': {'duration': transition_time}}],
                       'label': str(k), 'method': 'animate'} for k in range(101)]
        }]

        fig.update_layout(
            sliders=sliders,
            scene=dict(
                xaxis=dict(range=[-self.show_range, self.show_range], autorange=False),
                yaxis=dict(range=[-self.show_range, self.show_range], autorange=False),
                zaxis=dict(range=[-self.show_range, self.show_range], autorange=False),
                aspectmode='cube'
            ),
            title="Satellite Trajectory Visualization",
            updatemenus=[{
                'buttons': [
                    {'args': [None, {'frame': {'duration': duration_time, 'redraw': True},
                                     'fromcurrent': True, 'transition': {'duration': transition_time}}],
                     'label': 'Start Fall',
                     'method': 'animate'},
                    {'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate', 'transition': {'duration': 0}}],
                     'label': 'Pause',
                     'method': 'animate'}
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        fig.show()

def polar_plot(states1, states2):
    rho_from_states1 = [state[0] for state in states1]
    polar_from_states1 = [state[2] for state in states1]
    rho_from_states2 = [state[0] for state in states2]
    polar_from_states2 = [state[2] for state in states2]

    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.plot(range(len(states1)), rho_from_states1, label='Real Distance', marker='o')
    ax1.plot(range(len(states1)), rho_from_states2, label='Predicted Distance', linestyle='--')
    ax1.set_title('Distance Between Satellite And The Origin Of The Earth Over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(gs[1], projection='polar')
    rad = np.array(polar_from_states1)
    R = np.array(rho_from_states1)
    rad2 = np.array(polar_from_states2)
    R2 = np.array(rho_from_states2)
    rad3 = rad2
    R3 = R2

    colors = plt.cm.rainbow(np.linspace(0, 1, len(R2)))
    ax2.plot(rad, R, c='b', linestyle="dashed", label='Real Trajectory')
    ax2.plot(rad3, R3, c='g', linestyle="dashed", label='Predicted Trajectory')
    ax2.set_title('Polar Plot Over Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main(use_real_data=True):
    if use_real_data:
        main = Main(200)
        main.simulate()
        main.predict()
        R, rad, R2, rad2, R3, rad3 = main.output()
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.plot(rad, R - 6300000, c='b')
        #
        # plt.show()

        trd = [np.pi / 2 - config['satellite']['initial_conditions']['polar_angle']] * len(R)
        real_states_earth_cord = np.array([utilities.spherical_to_spherical(np.array([R, rad, trd]).T[i]) for i in range(len(R))])
        real_x, real_y, real_z = utilities.earth_to_xyz_bulk(real_states_earth_cord).T

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # plot = ax.scatter(real_x, real_y, real_z)
        # plt.show()
        states1 = list(zip(real_x, real_y, real_z))
        '''TODO real Predict Data'''
        states2 = list(zip(real_x, real_y, real_z))
    else:
        orbit_radius = Earth().re + 1000000
        theta = np.linspace(0, 2 * np.pi, 100)
        states1 = [(orbit_radius * np.cos(t), orbit_radius * np.sin(t), 0) for t in theta]

        orbit_radius = Earth().re + 1500000
        states2 = [(orbit_radius * np.cos(t), orbit_radius * np.sin(t), 0) for t in theta]


    visual_plotly = VisualisationPlotly(states1, states2)
    visual_plotly.visualise()

    polar_real_state = []
    for state in states1:
        polar_real_state.append(utilities.c_to_p(state))

    polar_predict_state = []
    for state in states2:
        polar_predict_state.append(utilities.c_to_p(state))

    polar_plot(polar_real_state, polar_predict_state)


main(False)
