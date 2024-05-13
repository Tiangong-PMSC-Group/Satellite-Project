import subprocess
import tkinter as tk
from config import config, save_config

import tkinter as tk
from tkinter import messagebox

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def on_button_click():
    try:
        # Collect input values
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

        # Validate inputs
        # Add your validation logic here if needed

        # Update configuration
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

        save_config(config)

        # Run simulation script if needed
        button.config(text="Simulation running...", fg='red')
        root.update()

        subprocess.run(["python", "3DVisual.py"], check=True)
        # subprocess.run(["python", "main.py"], check=True)
        # Update button state
        button.config(text="Finished, please check the display window.")
        root.update()

        # Close window after 3 seconds
        root.after(3000, root.destroy)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return


root = tk.Tk()

# root.geometry("400x300")  

# root.resizable(width=False, height=False)
label_left_pad = 0

i = 0
# label_radar = tk.Label(root, text="radar ", justify="left", anchor="w")
# label_radar.grid(row=i, column=0, padx=label_left_pad, pady=(0, 0), sticky='w')


# i=i+1

label1 = tk.Label(root, text="number of radars:", justify="left", anchor="w")
label1.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_radar = tk.Entry(root)
input_radar.insert(0, config['radar']['counts'])
input_radar.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))  

# i = i + 1
# label_radar_hint = tk.Label(root, text="hint arfgbaieufbiwraubgiuwuefiawuebfiauw eriguieruygueriytgf", justify="left", anchor="w")
# label_radar_hint.grid(row=i, column=0, columnspan=2, padx=10, pady=(0, 0), sticky='w')


i=i+1

label2 = tk.Label(root, text="satellite mass:", justify="left", anchor="w")
label2.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_satellite_mass = tk.Entry(root)
input_satellite_mass.insert(0, config['satellite']['mass'])
input_satellite_mass.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="satellite cross section:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_satellite_area = tk.Entry(root)
input_satellite_area.insert(0, config['satellite']['area'])
input_satellite_area.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="satellite drag coefficient:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_satellite_drag = tk.Entry(root)
input_satellite_drag.insert(0, config['satellite']['drag_coefficient'])
input_satellite_drag.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="initial satellite distance:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_satellite_distance = tk.Entry(root)
input_satellite_distance.insert(0, config['satellite']['initial_conditions']['distance'])
input_satellite_distance.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="initial satellite polar angle:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_satellite_angle = tk.Entry(root)
input_satellite_angle.insert(0, config['satellite']['initial_conditions']['polar_angle'])
input_satellite_angle.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="time interval(s):", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_time_interval = tk.Entry(root)
input_time_interval.insert(0, config['sim_config']['dt']['main_dt'])
input_time_interval.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="kalman frequency:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_kalman_frequency = tk.Entry(root)
input_kalman_frequency.insert(0, config['sim_config']['dt']['kalman_freq'])
input_kalman_frequency.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="radar frequency:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_radar_frequency = tk.Entry(root)
input_radar_frequency.insert(0, config['sim_config']['dt']['radar_freq'])
input_radar_frequency.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="radar noise for distance:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_radar_noise_distance = tk.Entry(root)
input_radar_noise_distance.insert(0, config['radar']['noise']['rho'])
input_radar_noise_distance.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="radar noise for polar:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_radar_noise_polar = tk.Entry(root)
input_radar_noise_polar.insert(0, config['radar']['noise']['theta'])
input_radar_noise_polar.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1
label3 = tk.Label(root, text="radar noise for azimuth:", justify="left", anchor="w")
label3.grid(row=i, column=0, padx=10, pady=(0, 0), sticky='w')
input_radar_noise_azimuth = tk.Entry(root)
input_radar_noise_azimuth.insert(0, config['radar']['noise']['phi'])
input_radar_noise_azimuth.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))

i=i+1

button = tk.Button(root, text="submit", command=on_button_click)
button.grid(row=i, column=0, columnspan=2, pady=5)


root.mainloop()