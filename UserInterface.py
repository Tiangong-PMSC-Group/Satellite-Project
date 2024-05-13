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
    # Collect input values
    radar_counts = input_radar.get()
    satellite_mass = input_satellite_mass.get()
    satellite_area = input_satellite_area.get()
    satellite_drag = input_satellite_drag.get()
    satellite_distance = input_satellite_distance.get()

    # Validate inputs
    if not radar_counts.isdigit() or not all(is_float(x) for x in [satellite_mass, satellite_area, satellite_drag, satellite_distance]):
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")
        return
    if any(float(x) <= 0 for x in [satellite_mass, satellite_area, satellite_drag, satellite_distance]):
        messagebox.showerror("Invalid Input", "Values must be positive numbers.")
        return

    # Update configuration
    config['radar']['counts'] = int(radar_counts)
    config['satellite']['mass'] = float(satellite_mass)
    config['satellite']['area'] = float(satellite_area)
    config['satellite']['drag_coefficient'] = float(satellite_drag)
    config['satellite']['initial_conditions']['distance'] = float(satellite_distance)
    save_config(config)

    # Update button state
    button.config(text="simulation running...", fg='red')
    root.update()

    # Run simulation script
    subprocess.run(["python", "3DVisual.py"], check=True)

    # Finalize
    button.config(text="finished, please check the display window.")
    root.update()
    root.after(3000, root.destroy)



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

button = tk.Button(root, text="submit", command=on_button_click)
button.grid(row=i, column=0, columnspan=2, pady=5)


root.mainloop()