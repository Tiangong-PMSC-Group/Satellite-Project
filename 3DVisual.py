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
        """Creates an instance of the Visualisation plot class.

        Args: 
            states1 (numpy.array): states of the satellite
            states2 (numpy.array): states of the radars data
            radar_positions (numpy.array): positions of the radatrs
        """

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
        """Creates the Earth graph.

        Args: 
            texture (numpy.array): surface texture of the Earth 

        Returns:
            float: texture's x coords
            float: texture's y coords
            float: texture's z coords
        """
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
        """Creates the Earth surface image.

        Returns:
            Surface Object: updates the surface with image
        """

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
        """Shows last point of impact for Kalman and ODE.

        Args:
            states (numpy.array): predicted and simulated states
            color (HEX / string): color of the last point
            name (string): name of the last point

        Returns:
            None
                or
            Scatter Object: plots a point at specific locations
        """
        # Check if there are enough states to highlight the last point
        if len(states) > 1:
            # Get the coordinates of the last state
            x, y, z = zip(*states[-1:])
            return go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(color=color, size=8, symbol='circle', opacity=0.8),
                name=name
            )
        return None

    # Create trajectory animation data
    def create_trajectory_frames(self, states1, states2, color1, color2):
        """Creates discretised frame of the predicted and simulated paths.

        Args:
            states1 (numpy.array): states of the satellite predictions
            states2 (numpy.array): states of the satellite simulations
            color1 (HEX / string): color of the predictions
            color2 (HEX / string): color of the simulations

        Returns:
            Frame Obejects: frames to be displayed
        """
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
                go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', line=dict(color=color1, width=7), opacity=0.7,
                             name="real trace        "),
                go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', line=dict(color=color2, width=7, dash='dash'), opacity=0.9,
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
        """Visualising function for centralised calling
        """
        # Create a new figure for visualization
        fig = go.Figure()

        # Add Earth's surface to the figure
        earth_surface = self.create_earth_surface()
        fig.add_trace(earth_surface)

        # Add empty traces for real and predicted satellite trajectories
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=7), opacity=0.7,
                                   name='Trace 1'))
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=7), opacity=0.8, name='Trace 2'))

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
        fig.frames = self.create_trajectory_frames(self.states1, self.states2, 'red', 'green')

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
def polar_plot(true_states_earth_coord, predicted_states_earth_coord, radar_states_earth_cord, R, rad, R2, rad2, rad3, var_r, var_phi):
    """Generates a plot in polar coordinates.

    Args:
        true_states_earth_coord (numpy.array): True states of the satellite in Earth coordinates.
        predicted_states_earth_coord (numpy.array): Predicted states of the satellite in Earth coordinates.
        radar_states_earth_cord (numpy.array): Radar-detected states of the satellite in Earth coordinates.
        R (numpy.array): True radius of the orbit at each time step.
        rad (numpy.array): True angles of the orbit at each time step.
        R2 (numpy.array): Predicted radius of the orbit at each time step.
        rad2 (numpy.array): Predicted angles of the orbit at each time step.
        rad3 (numpy.array): Angles corresponding to radar measurements.
        var_r (numpy.array): Variance of the radius from the Kalman filter at each time step.
        var_phi (numpy.array): Variance of the angle from the Kalman filter at each time step.
    """
    earth = Earth()

    pred_heights = np.zeros(len(predicted_states_earth_coord))
    true_heights = np.zeros(len(true_states_earth_coord))
    radar_heights = np.zeros(len(radar_states_earth_cord))

    Vp = var_phi
    Vr = var_r

    for i in range(len(pred_heights)):
        pred_heights[i] = earth.distance_to_surface(predicted_states_earth_coord[i])['distance']

    for i in range(len(true_heights)):
        true_heights[i] = earth.distance_to_surface(true_states_earth_coord[i])['distance']

    for i in range(len(radar_heights)):
        radar_heights[i] = earth.distance_to_surface(radar_states_earth_cord[i])['distance']

    fig = plt.figure(constrained_layout=True, figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], width_ratios = [3, 2])

    # Plotting the distance over time
    ax1 = plt.subplot(gs[0,0])
    ax1.plot(range(len(R2)), R2, color = 'dodgerblue', label='Predicted Radius of Orbit',linewidth = 3)
    ax1.plot(range(len(R)), R, color = 'black',  alpha = 0.8, linestyle = 'dashed', label='Real Radius of Orbit', linewidth = 3)

    ax1.set_title('Radius of Orbit for each Time Steps')
    ax1.set_xlabel('Number of Time Steps')
    ax1.set_ylabel('Distance (m)')
    ax1.legend(loc='lower left')
    ax1.grid(True)

    # Plotting the polar and distance
    ax2 = plt.subplot(gs[0,1], projection = 'polar')
    

    # Plot the traces
    ax2.plot(rad2, pred_heights, c='dodgerblue', label='Predicted Trajectory', linewidth=3)
    ax2.plot(rad, true_heights, c='black', linestyle="dashed",  alpha = 0.7, label='Real Trajectory', linewidth=3)

    ax2.set_title('Polar Angle (rad) vs Altitude (m)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position


    ax4 = plt.subplot(gs[1,0])
    ax4.semilogy(range(len(Vr)), Vr, color = 'crimson', label = 'Variance of Radius (m$^2$)', linewidth = 3)
    ax4.semilogy(range(len(Vp)), Vp, color = 'darkorange', label = 'Variance of Polar Angle (rad$^2$)', linewidth = 3)
    ax4.legend()
    ax4.grid()
    ax4.set(title='Kalman Filter Variance for each Time Step', xlabel = 'Number of Time Steps', ylabel = 'log(variance)')


    ax3 = plt.subplot(gs[1,1], projection= 'polar')

    # Plot the traces
    ax3.plot(rad, true_heights, c='black', linestyle="dashed", label='Real Trajectory', linewidth=3, zorder = -3)
    ax3.scatter(rad3, radar_heights, c='g', label='Radar Data', s = 5, zorder = -2, alpha = 0.3)
    ax3.set_zorder(-1)

    ax3.set_title('Polar Angle (rad) vs Altitude (m)')
    ax3.legend(loc='upper left',  bbox_to_anchor=(1, 1))  # Adjust legend position

    data_tab = [config['satellite']['initial_conditions']['distance'], config['satellite']['initial_conditions']['polar_angle'],
        str(config['radar']['counts']), config['radar']['noise']['rho'], config['radar']['noise']['theta'], config['Kalman']['initial_r_guess'],
        config['Kalman']['initial_angle_guess'], config['Kalman']['initial_vr_guess'], config['Kalman']['initial_vphi_guess'],
        config['satellite']['mass'], config['satellite']['area'], config['satellite']['drag_coefficient'], config['sim_config']['dt']['main_dt'],
        str(config['sim_config']['dt']['radar_freq'])]

    titles = ['Satellite\nDistance (m)', 'Satellite\nAngle (rad)', 'Radar\nCounts','Distance\nNoise', 'Polar\nNoise', 'Prior\nDistance (m)', 
              'Prior\nAngle (rad)', r'Prior $v_r$', r'Prior $v_\phi$', 'Satellite\nMass (kg)', 'Satellite\nArea'r'(m$^2$)', 'Satellite\nDrag', 'Time\nInterval', 'Radar\nFrequency' ]
    cell_text = [[f'{x}' for x in data_tab]]

    ax5 = plt.subplot(gs[2,:])
    tab = ax5.table(cellText = cell_text, colLabels = titles, cellLoc= 'center', loc = 'center', colColours = ['lightgrey' for i in range(len(titles))])
    tab.scale(1,2)
    tab.auto_set_font_size(False)
    tab.set_fontsize=(30)
    ax5.axis('off')

    plt.suptitle('Simulation and Prediction Results', fontsize = 'xx-large', fontweight = 'bold')
    plt.show()


def main():
    """Generates the simulation and its environment.
    """
    # Run simulation
    main = Main()
    main.simulate()
    main.predict()
    # Get distance and polar datas from real simulator, predictor and radar
    R, rad, R2, rad2, R3, rad3, var_r, var_phi = main.output()

    # Transform the data to the appropriate coordinate
    trd = [np.pi / 2 - config['satellite']['initial_conditions']['polar_angle']] * len(R)
    real_states_earth_cord = np.array(
        [utilities.spherical_to_spherical(np.array([R, rad, trd]).T[i]) for i in range(len(R))])
    real_x, real_y, real_z = utilities.earth_to_xyz_bulk(real_states_earth_cord).T

    trd = [np.pi / 2 - config['satellite']['initial_conditions']['polar_angle']] * len(R2)
    predict_states_earth_cord = np.array(
        [utilities.spherical_to_spherical(np.array([R2, rad2, trd]).T[i]) for i in range(len(R2))])
    pred_x, pred_y, pred_z = utilities.earth_to_xyz_bulk(predict_states_earth_cord).T

    trd = [np.pi / 2 - config['satellite']['initial_conditions']['polar_angle']] * len(R3)
    radar_states_earth_cord = np.array(
        [utilities.spherical_to_spherical(np.array([R3, rad3, trd]).T[i]) for i in range(len(R3))])
    
    states1 = list(zip(real_x, real_y, real_z))
    states2 = list(zip(pred_x, pred_y, pred_z))
    radar_system = main.BACC

    # Show 3D earth and trajectory
    visual_plotly = VisualisationPlotly(states1, states2,radar_system.get_radar_positions())
    visual_plotly.visualise()

    # Show static plots of trajectories
    polar_plot(real_states_earth_cord, predict_states_earth_cord, radar_states_earth_cord, R, rad, R2, rad2, rad3, var_r, var_phi)


main()
