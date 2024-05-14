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

# 3D Visualization
class VisualisationPlotly:
    def __init__(self, states1, states2, radar_positions):
        self.states1 = states1
        self.states2 = states2
        self.radar_positions = []
        for pos in radar_positions:
            self.radar_positions.append(utilities.p_to_c(pos))
        self.re = Earth().re
        self.rp = Earth().rp
        self.show_range = 8000000

    # Create Earth Sphere Data
    def sphere(self, texture):
        # Get the number of latitude and longitude points
        N_lat = int(texture.shape[0])
        N_lon = int(texture.shape[1])

        # Create arrays for theta (azimuthal angle) and phi (polar angle)
        theta = np.linspace(0, 2 * np.pi, N_lat)
        phi = np.linspace(0, np.pi, N_lon)

        # Calculate the Cartesian coordinates for the sphere surface
        x0 = self.re * np.outer(np.cos(theta), np.sin(phi))
        y0 = self.re * np.outer(np.sin(theta), np.sin(phi))
        z0 = self.rp * np.outer(np.ones(N_lat), np.cos(phi))

        return x0, y0, z0

    # Create Earth Surface Data, Utilizing Real Earth Imagery
    def create_earth_surface(self):
        # Define the color scale for the Earth's surface
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

        # Load and resize the texture image for the Earth
        texture = np.asarray(Image.open('earth.jpg').resize((200, 200))).T
        x, y, z = self.sphere(texture)  # Generate sphere coordinates
        return go.Surface(x=x, y=y, z=z, surfacecolor=texture, colorscale=colorscale, opacity=0.85, showscale=False)

    # Find the position where satellite hit the earth
    def highlight_last_points(self, states, color, name):
        # Check if there are enough states to highlight the last point
        if len(states) > 1:
            # Get the coordinates of the last state
            x, y, z = zip(*states[-1:])
            return go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(color=color, size=5, symbol='circle', opacity=0.5),
                name=name
            )
        return None

    # Create trajectory animation data
    def create_trajectory_frames(self, states1, states2, color1, color2):
        frames = []
        # Iterate through 101 progress points to create frames(0-100)
        for progress in range(101):
            # Calculate the index up to which the states should be included based on progress
            progress_index = int(len(states1) * progress / 100)
            # Unpack the states up to the progress index for both trajectories
            x1, y1, z1 = zip(*states1[:progress_index]) if progress_index > 0 else ([], [], [])
            x2, y2, z2 = zip(*states2[:progress_index]) if progress_index > 0 else ([], [], [])

            # Create frame data of trajectorys
            frame_data = [
                go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', line=dict(color=color1, width=4), opacity=0.7,
                             name="real trace        "),
                go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', line=dict(color=color2, width=4, dash='dash'), opacity=0.7,
                             name="predicted trace       "),
            ]

            # Highlight the location of satellite hit on the earth the progress is at 100%
            if progress == 100:
                last_point1 = self.highlight_last_points(states1, color1, "Real crash Pos")
                last_point2 = self.highlight_last_points(states2, color2, "Predicted crash Pos")
                if last_point1:
                    frame_data.append(last_point1)
                if last_point2:
                    frame_data.append(last_point2)
            else:
                # Otherwise add a placeholder trace.
                frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers'))
                frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers'))
            # Add the frame to the list of frames
            frames.append(go.Frame(data=frame_data, name=str(progress), traces=[1, 2, 3, 4]))

        return frames

    # Show the 3D earth and trajectory
    def visualise(self):
        # Create a new figure for visualization
        fig = go.Figure()

        # Add Earth's surface to the figure
        earth_surface = self.create_earth_surface()
        fig.add_trace(earth_surface)

        # Add empty traces for real and predicted satellite trajectories
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='yellow', width=6), opacity=0.8,
                                   name='Trace 1'))
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=6), opacity=0.8, name='Trace 2'))

        # Add empty markers for real and predicted crash positions
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Real crash Pos'))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Predicted crash Pos'))

        # Add radar positions to the figure
        radar_x, radar_y, radar_z = zip(*self.radar_positions)
        fig.add_trace(go.Scatter3d(
            x=radar_x, y=radar_y, z=radar_z, mode='markers',
            marker=dict(color='pink', size=2, symbol='diamond', opacity=0.8),
            name='Radar Positions'
        ))

        # Create frames for the animation of trajectories
        fig.frames = self.create_trajectory_frames(self.states1, self.states2, 'yellow', 'red')

        # Define the slider for the animation
        sliders = [{
            'pad': {"t": 30},
            'steps': [{'args': [[str(k)], {'frame': {'duration': duration_time, 'redraw': True},
                                           'mode': 'immediate', 'transition': {'duration': transition_time}}],
                       'label': str(k), 'method': 'animate'} for k in range(101)]
        }]

        # Update the layout of the figure
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
                'pad': {'r': 10, 't': 110},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        # Display the figure
        fig.show()


# Plot static plots of the trajectories
def polar_plot(states1, states2):
    rho_from_states1 = [state[0] for state in states1]
    polar_from_states1 = [state[2] for state in states1]
    rho_from_states2 = [state[0] for state in states2]
    polar_from_states2 = [state[2] for state in states2]

    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # Plotting the distance over time
    ax1 = plt.subplot(gs[0])
    ax1.plot(range(len(states1)), rho_from_states1, label='Real Distance', marker='o')
    ax1.plot(range(len(states2)), rho_from_states2, label='Predicted Distance', linestyle='--')
    ax1.set_title('Distance Between Satellite And The Origin Of The Earth Over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Distance (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plotting the polar and distance
    ax2 = plt.subplot(gs[1], projection='polar')
    rad = np.array(polar_from_states1)
    R = np.array(rho_from_states1)
    rad2 = np.array(polar_from_states2)
    R2 = np.array(rho_from_states2)

    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.tick_params(axis='both', which='both', colors='gray', width=0.5)

    # Plot the traces
    ax2.plot(rad, R, c='b', linestyle="solid", label='Real Trajectory', linewidth=2)
    ax2.plot(rad2, R2, c='r', linestyle="dashed", label='Predicted Trajectory', linewidth=2)

    ax2.set_title('Polar Plot Over Time')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.8, 1))  # Adjust legend position

    plt.tight_layout()
    plt.show()


def main():
    # Run simulation
    main = Main(200)
    main.simulate()
    main.predict()
    # Get distance and polar datas from real simulator, predictor and radar
    R, rad, R2, rad2, R3, rad3 = main.output()

    # Transform the data to the appropriate coordinate
    trd = [np.pi / 2 - config['satellite']['initial_conditions']['polar_angle']] * len(R)
    real_states_earth_cord = np.array(
        [utilities.spherical_to_spherical(np.array([R, rad, trd]).T[i]) for i in range(len(R))])
    real_x, real_y, real_z = utilities.earth_to_xyz_bulk(real_states_earth_cord).T

    trd = [np.pi / 2 - config['satellite']['initial_conditions']['polar_angle']] * len(R2)
    predict_states_earth_cord = np.array(
        [utilities.spherical_to_spherical(np.array([R2, rad2, trd]).T[i]) for i in range(len(R2))])
    pred_x, pred_y, pred_z = utilities.earth_to_xyz_bulk(predict_states_earth_cord).T

    states1 = list(zip(real_x, real_y, real_z))
    states2 = list(zip(pred_x, pred_y, pred_z))
    radar_system = main.BACC

    # Show 3D earth and trajectory
    visual_plotly = VisualisationPlotly(states1, states2,radar_system.get_radar_positions())
    visual_plotly.visualise()

    polar_real_state = []
    for state in states1:
        polar_real_state.append(utilities.c_to_p(state))

    polar_predict_state = []
    for state in states2:
        polar_predict_state.append(utilities.c_to_p(state))

    # Show static plots of trajectories
    polar_plot(polar_real_state, polar_predict_state)


main()
