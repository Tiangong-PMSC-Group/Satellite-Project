import subprocess
import tkinter as tk
from tkinter import messagebox
import psutil  # Add this import to check for running processes
from config import config, config_in, save_config, load_config

def on_button_click():
    """Handle button click event, validate input, update config, and run scripts."""
    try:
        file_number = input_file.get()
        print(file_number)
        if file_number == str(0):
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
        else:
            path = 'config_' + str(file_number) + '.json'
            config_i = load_config(path)

            radar_counts = str(config_i['radar']['counts'])
            satellite_mass = config_i['satellite']['mass']
            satellite_area = config_i['satellite']['area']
            satellite_drag = config_i['satellite']['drag_coefficient']
            satellite_distance = config_i['satellite']['initial_conditions']['distance']
            satellite_angle = config_i['satellite']['initial_conditions']['polar_angle']
            time_interval = config_i['sim_config']['dt']['main_dt']
            kalman_frequency = str(config_i['sim_config']['dt']['kalman_freq'])
            radar_frequency = str(config_i['sim_config']['dt']['radar_freq'])
            radar_noise_distance = config_i['radar']['noise']['rho']
            radar_noise_polar = config_i['radar']['noise']['theta']
            radar_noise_azimuth = config_i['radar']['noise']['phi']

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

        # Close previous graph window if it exists
        close_previous_instance("3DVisual.py")

        # subprocess.run(["python", "run_main.py"], check=True)
        subprocess.run(["python", "3DVisual.py"], check=True)

        # Update button state after scripts run
        button.config(text="Finished, please check the display window.")
        root.update()

        # Reset input fields
        #reset_input_fields()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    
    finally:
        # Reset input fields and button state
        reset_input_fields()
        button.config(text="Start", fg='black')
        root.update()

def close_previous_instance(script_name):
    """Close any previous instances of the given script."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and script_name in cmdline:
            proc.terminate()
            proc.wait()


def reset_input_fields():
    """Reset all input fields to their default values."""
    input_file.delete(0, tk.END)
    input_file.insert(0, "0")

    input_radar.delete(0, tk.END)
    input_radar.insert(0, config_in['radar']['counts'])

    input_satellite_mass.delete(0, tk.END)
    input_satellite_mass.insert(0, config_in['satellite']['mass'])

    input_satellite_area.delete(0, tk.END)
    input_satellite_area.insert(0, config_in['satellite']['area'])

    input_satellite_drag.delete(0, tk.END)
    input_satellite_drag.insert(0, config_in['satellite']['drag_coefficient'])

    input_satellite_distance.delete(0, tk.END)
    input_satellite_distance.insert(0, config_in['satellite']['initial_conditions']['distance'])

    input_satellite_angle.delete(0, tk.END)
    input_satellite_angle.insert(0, config_in['satellite']['initial_conditions']['polar_angle'])

    input_time_interval.delete(0, tk.END)
    input_time_interval.insert(0, config_in['sim_config']['dt']['main_dt'])

    input_kalman_frequency.delete(0, tk.END)
    input_kalman_frequency.insert(0, config_in['sim_config']['dt']['kalman_freq'])

    input_radar_frequency.delete(0, tk.END)
    input_radar_frequency.insert(0, config_in['sim_config']['dt']['radar_freq'])

    input_radar_noise_distance.delete(0, tk.END)
    input_radar_noise_distance.insert(0, config_in['radar']['noise']['rho'])

    input_radar_noise_polar.delete(0, tk.END)
    input_radar_noise_polar.insert(0, config_in['radar']['noise']['theta'])

    input_radar_noise_azimuth.delete(0, tk.END)
    input_radar_noise_azimuth.insert(0, config_in['radar']['noise']['phi'])

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
    inputs = [("use default config file?", 0),
              ("number of radars:", config_in['radar']['counts']),
              ("satellite mass:", config_in['satellite']['mass']),
              ("satellite cross section:", config_in['satellite']['area']),
              ("satellite drag coefficient:", config_in['satellite']['drag_coefficient']),
              ("initial satellite distance:", config_in['satellite']['initial_conditions']['distance']),
              ("initial satellite polar angle:", config_in['satellite']['initial_conditions']['polar_angle']),
              ("time interval(s):", config_in['sim_config']['dt']['main_dt']),
              ("kalman frequency:", config_in['sim_config']['dt']['kalman_freq']),
              ("radar frequency:", config_in['sim_config']['dt']['radar_freq']),
              ("radar noise for distance:", config_in['radar']['noise']['rho']),
              ("radar noise for polar:", config_in['radar']['noise']['theta']),
              ("radar noise for azimuth:", config_in['radar']['noise']['phi'])
              ]

    # Create label and entry widgets for each input
    entries = []
    for i, (label_text, default_value) in enumerate(inputs):
        entry = create_label_entry(root, label_text, i, default_value)
        entries.append(entry)

    # Unpack entries into individual variables for easier access
    (input_file, input_radar, input_satellite_mass, input_satellite_area, input_satellite_drag,
     input_satellite_distance, input_satellite_angle, input_time_interval,
     input_kalman_frequency, input_radar_frequency, input_radar_noise_distance,
     input_radar_noise_polar, input_radar_noise_azimuth) = entries

    # Create and grid the Start button
    button = tk.Button(root, text="Start", command=on_button_click)
    button.grid(row=len(inputs), column=0, columnspan=2, pady=5)

    root.mainloop()
