# TIANGONG

TIANGONG is a Python application that utilizes an Extended Kalman Filter to predict dynamics of a satellite de-orbital using simulated data and ground-based radar stations (within line of sight) that provide positional data as the satellite's orbit decays due to atmospheric drag. A three-dimensional non-spherical earth, equipped with US Standard atmosphere consisting of 34 layers, non-equatorial orbits as well as non-equidistant radar stations with LOS (line of sight) functionality are implemented to make the modeling and prediction more realistic.

## Installation

To install and set up the program, follow these steps:

1. **Check Python Installation:**
   - Ensure you have Python installed on your machine, which you can download from the [official Python website](https://www.python.org/).

2. **Download Program Files:**
   - Obtain all the program files from Moodle.

3. **Locate and Open Files:**
   - Locate the downloaded files on your computer.

4. **Execute Using Command Line or Code Editor:**
   - Open the executable using command line or code editor of your choice.

5. **Install Dependencies:**
   - To ensure proper installation and functionality of the files, make sure you have the following dependencies installed:
     - ipython version 8.18.1 or higher
     - ipywidgets version 8.0.4 or higher
     - matplotlib version 3.8.4 or higher
     - numpy version 1.24.3 or higher
     - Pillow version 10.3.0 or higher
     - plotly version 5.9.0 or higher
     - scipy version 1.11.1 or higher
     - psutil version 5.8.0 or higher
     - tk version 0.1.0 or higher

The program automatically checks for and installs the required libraries, but these are mentioned here for completeness.

## Implementation

Run the Tiangong.py script, it will launch a user-friendly dashboard with default parameter conditions, allowing modification of initial conditions, noise parameters, the number of radars, and Kalman filter frequency. The dashboard also features a default setting which can be accessed using a default configuration file if available. The config file should follows the following convention name "config_{name}" and at the dashboard is accessed by type just "name". After configuring the desired parameters, click the Start button on the dashboard to commence the simulation, with completion typically taking a few minutes depending on your device's computational power. Upon completion, a figure will display on the screen, including a plot of distance versus time and a polar 2D plot depicting changes in radial distance and angle over time, encompassing both true and predicted dynamics. Additionally, a web page will open in the default local browser. The animation features a comprehensive 3D plot that visualizes the trajectory over time, allowing manual adjustments for better analysis and rotation of the earth for enhanced visuals, highlighting both real and predicted dynamics and indicating which radars are within the line of sight of the satellite at multiple discrete time steps. The program does not need to be rerun to conduct a new simulation; simply close the graph window, and the button resets to "Start" again. Further elaboration is available in the tutorial document which has additional guidance and images for clarity.

Default values are provided on the dashboard as a starting point, and these can be changed within reasonable values and assertion errors are built into the system to indicate possible errors in input.
