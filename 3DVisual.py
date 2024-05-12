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
        return go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale='Blues',showscale=False)

    def create_trajectory(self, states, color, progress=100, opacity=1.0):
        progress_index = int(len(states) * progress / 100)
        x, y, z = zip(*states[:progress_index])
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color=color, width=2),
            opacity=opacity,
            name="Real Satellite Trajectory"
        )

    def highlight_last_points(self, states, color):
        # Extract last two points
        if len(states) > 1:
            x, y, z = zip(*states[-1:])
            return go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(color=color, size=5, symbol='circle',opacity=0.5 ),
                name="Final Location"
            )
        return None

    def visualise(self, progress=100):
        fig = go.Figure()
        fig.add_trace(self.create_earth_surface())
        fig.add_trace(self.create_trajectory(self.states1, 'red', progress, opacity=0.8))
        fig.add_trace(self.create_trajectory(self.states2, 'green', progress, opacity=0.6))
        fig.add_trace(self.highlight_last_points(self.states1, 'red'))
        fig.add_trace(self.highlight_last_points(self.states2, 'green'))

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-self.show_range, self.show_range], autorange=False),
                yaxis=dict(range=[-self.show_range, self.show_range], autorange=False),
                zaxis=dict(range=[-self.show_range, self.show_range], autorange=False),
                aspectmode='cube'
            ),
            title="Satellite Trajectory Visualization"
        )
        fig.show()



orbit_radius = Earth().re + 1000000
theta = np.linspace(0, 2 * np.pi, 100)
states1 = [(orbit_radius * np.cos(t), orbit_radius * np.sin(t), 0) for t in theta]
states2 = [(orbit_radius * np.cos(t), orbit_radius * np.sin(t), 0) for t in theta]

polar_real_state = []
for state in states1:
    polar_real_state.append(utilities.c_to_p(state))

polar_predict_state = []
for state in states2:
    polar_predict_state.append(utilities.c_to_p(state))

visual_plotly = VisualisationPlotly(states1, states2)
interact(visual_plotly.visualise, progress=IntSlider(min=0, max=100, step=1, value=100, description='Progress'))

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

# polar_plot(polar_real_state, polar_predict_state)