import subprocess
import tkinter as tk
from tkinter import messagebox
from config import config, save_config

def on_button_click():
    """Handle button click event, validate input, update config, and run scripts."""
    try:
        # Collect input values from entries
        radar_counts = input_radar.get()
        satellite_mass = input_satellite_mass.get()
        satellite_area = input_satellite_area.get()
        satellite_drag = input_satellite_drag.get()
        satellite_distance = input_satellite_distance.get()
        satellite_angle = input_satellite_angle.get()
        time_interval = input_time_interval.get()
        kalman_frequency = input_kalman_frequency.get()
        radar_frequency = input_radar_frequency.get()
        radar_noise_distance = input_radar_noise_distance.get()
        radar_noise_polar = input_radar_noise_polar.get()
        radar_noise_azimuth = input_radar_noise_azimuth.get()

        # Check inputs
        if not radar_counts.isdigit():
            messagebox.showerror("Input Error", "Radar counts must be a whole number.")
            return
        if not is_float(satellite_mass):
            messagebox.showerror("Input Error", "Satellite mass must be a numeric value.")
            return
        if not is_float(satellite_area):
            messagebox.showerror("Input Error", "Satellite area must be a numeric value.")
            return
        if not is_float(satellite_drag):
            messagebox.showerror("Input Error", "Satellite drag coefficient must be a numeric value.")
            return
        if not is_float(satellite_distance):
            messagebox.showerror("Input Error", "Satellite distance must be a numeric value.")
            return
        if not is_float(satellite_angle):
            messagebox.showerror("Input Error", "Satellite angle must be a numeric value.")
            return
        if not is_float(time_interval):
            messagebox.showerror("Input Error", "Time interval must be a numeric value.")
            return
        if not kalman_frequency.isdigit():
            messagebox.showerror("Input Error", "Kalman frequency must be a whole number.")
            return
        if not radar_frequency.isdigit():
            messagebox.showerror("Input Error", "Radar frequency must be a whole number.")
            return
        if not is_float(radar_noise_distance):
            messagebox.showerror("Input Error", "Radar noise distance must be a numeric value.")
            return
        if not is_float(radar_noise_polar):
            messagebox.showerror("Input Error", "Radar noise polar must be a numeric value.")
            return
        if not is_float(radar_noise_azimuth):
            messagebox.showerror("Input Error", "Radar noise azimuth must be a numeric value.")
            return

        # Update configuration with validated values
        config['radar']['counts'] = int(radar_counts)
        config['satellite']['mass'] = float(satellite_mass)
        config['satellite']['area'] = float(satellite_area)
        config['satellite']['drag_coefficient'] = float(satellite_drag)
        config['satellite']['initial_conditions']['distance'] = float(satellite_distance)
        config['satellite']['initial_conditions']['polar_angle'] = float(satellite_angle)
        config['sim_config']['dt']['main_dt'] = float(time_interval)
        config['sim_config']['dt']['kalman_freq'] = int(kalman_frequency)
        config['sim_config']['dt']['radar_freq'] = int(radar_frequency)
        config['radar']['noise']['rho'] = float(radar_noise_distance)
        config['radar']['noise']['theta'] = float(radar_noise_polar)
        config['radar']['noise']['phi'] = float(radar_noise_azimuth)

        save_config(config)  # Save updated config

        # Run simulation scripts
        button.config(text="Simulation running...", fg='red')
        root.update()

        # subprocess.run(["python", "run_main.py"], check=True)
        subprocess.run(["python", "3DVisual.py"], check=True)

        # Update button state after scripts run
        button.config(text="Finished, please check the display window.")
        root.update()

        # Close window after 3 seconds
        root.after(3000, root.destroy)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def is_float(value):
    """Check if a value can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def create_label_entry(root, text, row, default_value):
    """Create a label and entry widget, insert default value, and grid them."""
    label = tk.Label(root, text=text, justify="left", anchor="w")
    label.grid(row=row, column=0, padx=10, pady=(0, 0), sticky='w')
    entry = tk.Entry(root)
    entry.insert(0, default_value)
    entry.grid(row=row, column=1, padx=(0, 10), pady=(0, 0))
    return entry


import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing requirements: {e}")

if __name__ == "__main__":
    install_requirements()
        
    root = tk.Tk()  # Create main application window

    # List of input labels and default values
    inputs = [
        ("number of radars:", config['radar']['counts']),
        ("satellite mass:", config['satellite']['mass']),
        ("satellite cross section:", config['satellite']['area']),
        ("satellite drag coefficient:", config['satellite']['drag_coefficient']),
        ("initial satellite distance:", config['satellite']['initial_conditions']['distance']),
        ("initial satellite polar angle:", config['satellite']['initial_conditions']['polar_angle']),
        ("time interval(s):", config['sim_config']['dt']['main_dt']),
        ("kalman frequency:", config['sim_config']['dt']['kalman_freq']),
        ("radar frequency:", config['sim_config']['dt']['radar_freq']),
        ("radar noise for distance:", config['radar']['noise']['rho']),
        ("radar noise for polar:", config['radar']['noise']['theta']),
        ("radar noise for azimuth:", config['radar']['noise']['phi'])
    ]

    # Create label and entry widgets for each input
    entries = []
    for i, (label_text, default_value) in enumerate(inputs):
        entry = create_label_entry(root, label_text, i, default_value)
        entries.append(entry)

    # Unpack entries into individual variables for easier access
    (input_radar, input_satellite_mass, input_satellite_area, input_satellite_drag,
    input_satellite_distance, input_satellite_angle, input_time_interval,
    input_kalman_frequency, input_radar_frequency, input_radar_noise_distance,
    input_radar_noise_polar, input_radar_noise_azimuth) = entries

    # Create and grid the Start button
    button = tk.Button(root, text="Start", command=on_button_click)
    button.grid(row=len(inputs), column=0, columnspan=2, pady=5)

    root.mainloop()
