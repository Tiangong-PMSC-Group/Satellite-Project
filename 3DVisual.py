import matplotlib
matplotlib.use('TkAgg')
import plotly.graph_objects as go
from Earth import Earth
import numpy as np
from ipywidgets import interact, IntSlider
import numpy as np
import matplotlib.pyplot as plt
from Earth import Earth

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
        return go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale='Blues')

    def create_trajectory(self, states, color, progress=100, opacity=1.0):
        progress_index = int(len(states) * progress / 100)
        x, y, z = zip(*states[:progress_index])
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color=color, width=2),
            opacity=opacity  # Set the opacity for the trajectory
        )

    def visualise(self, progress=100):
        fig = go.Figure()
        fig.add_trace(self.create_earth_surface())
        fig.add_trace(self.create_trajectory(self.states1, 'red', progress, opacity=0.8))
        fig.add_trace(self.create_trajectory(self.states2, 'green', progress, opacity=0.6))

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


visual_plotly = VisualisationPlotly(states1, states2)
interact(visual_plotly.visualise, progress=IntSlider(min=0, max=100, step=1, value=100, description='Progress'))


# plot figures
x_coords1 = [state[0] for state in states1]
x_coords2 = [state[0] for state in states2]

plt.figure(figsize=(10, 5))
plt.plot(theta, x_coords1, label='State 1 X Coordinate', marker='o')
plt.plot(theta, x_coords2, label='State 2 X Coordinate', linestyle='--')
plt.title('X Coordinates Over Time')
plt.xlabel('Time (radians)')
plt.ylabel('X Coordinate')
plt.legend()
plt.grid(True)
plt.show()
