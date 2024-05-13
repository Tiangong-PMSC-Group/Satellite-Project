import matplotlib

import utilities

matplotlib.use('TkAgg')
import plotly.graph_objects as go
from Earth import Earth
import numpy as np
from ipywidgets import interact, IntSlider
import numpy as np
import matplotlib.pyplot as plt
from Earth import Earth

import plotly.graph_objects as go
from Earth import Earth
import numpy as np

import plotly.graph_objects as go
from Earth import Earth
import numpy as np

transition_time = 10
duration_time = 10
class VisualisationPlotly:
    def __init__(self, states1, states2):
        self.states1 = states1
        self.states2 = states2
        self.re = Earth().re
        self.rp = Earth().rp
        self.show_range = 8000000

    def create_earth_surface(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.re * np.outer(np.cos(u), np.sin(v))
        y = self.re * np.outer(np.sin(u), np.sin(v))
        z = self.rp * np.outer(np.ones(np.size(u)), np.cos(v))
        return go.Surface(x=x, y=y, z=z, opacity=1, colorscale='Blues', showscale=False)

    def highlight_last_points(self, states, color):
        # Extract last point
        if len(states) > 1:
            x, y, z = zip(*states[-1:])
            return go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(color=color, size=5, symbol='circle', opacity=0.5),
                name="Final Location"
            )
        return None

    def create_trajectory_frames(self, states1, states2, color1, color2):
        frames = []
        earth_surface = self.create_earth_surface()
        for progress in range(101):
            progress_index = int(len(states1) * progress / 100)
            x1, y1, z1 = zip(*states1[:progress_index]) if progress_index > 0 else ([], [], [])
            x2, y2, z2 = zip(*states2[:progress_index]) if progress_index > 0 else ([], [], [])

            frame_data = [
                earth_surface,
                go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', line=dict(color=color1, width=2), opacity=0.5),
                go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', line=dict(color=color2, width=2), opacity=0.5)
            ]

            # At 100% progress, add last point highlights
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

            frames.append(go.Frame(data=frame_data, name=str(progress)))

        return frames

    def visualise(self):
        fig = go.Figure()

        # Initial plot to setup the space and static elements
        earth_surface = self.create_earth_surface()
        fig.add_trace(earth_surface)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=2), opacity=0.5))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=2), opacity=0.5))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers'))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers'))
        # Add frames for both trajectories
        fig.frames = self.create_trajectory_frames(self.states1, self.states2, 'red', 'green')

        # Setup sliders and buttons for animation control
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
                                       'mode': 'immediate',
                                       'transition': {'duration': 0}}],
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


# Use this class
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
def polar_plot(states1, states2):

    rho_from_states1 = [state[0] for state in states1]
    polar_from_states1 = [state[2] for state in states1]
    rho_from_states2 = [state[0] for state in states2]
    polar_from_states2 = [state[2] for state in states2]


    fig, axs = plt.subplots(2, 1, figsize=(10, 9))
    print(polar_from_states2)

    axs[0].plot(range(len(states1)), rho_from_states1, label='Real Distance', marker='o')
    axs[0].plot(range(len(states1)), rho_from_states2, label='Predicted Distance', linestyle='--')
    axs[0].set_title('Distance Between Satellite And The Origin Of The Earth Over Time')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Distance(m)')
    axs[0].legend()
    axs[0].grid(True)


    axs[1].plot(range(len(states1)), polar_from_states1, label='Polar Of Real Trajactory', marker='o')
    axs[1].plot(range(len(states1)), polar_from_states2, label='Polar Of Predicted Trajactory', linestyle='--')
    axs[1].set_title('Polar Over Time')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Polar')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

polar_plot(polar_real_state, polar_predict_state)