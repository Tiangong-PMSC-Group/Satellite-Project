import subprocess
import tkinter as tk
from config import config, save_config

def on_button_click():
    # for widget in root.winfo_children():
    #     if widget != button:
    #         widget.grid_remove()

    config['radar']['counts'] = int(input_radar.get())
    config['satellite']['mass'] = float(input_satellite_mass.get())
    config['satellite']['area'] = float(input_satellite_area.get())
    config['satellite']['drag_coefficient'] = float(input_satellite_drag.get())
    config['satellite']['initial_conditions']['distance'] = float(input_satellite_distance.get())
    save_config(config)

    button.config(text="simulation running...")
    button.config(fg='red')
    root.update()
    # process = subprocess.Popen(["python", "/Users/han/Documents/GitHub/Satellite-Project/Visualization.py"])
    # subprocess.Popen(["python", "/Users/han/Documents/GitHub/Satellite-Project/Visualization.py"])
    # subprocess.run(["python", "/Users/han/Documents/GitHub/Satellite-Project/Visualization.py"], check=True)
    subprocess.run(["python", "try.py"], check=True)
    subprocess.run(["python", "Visualization.py"], check=True)

    # subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute",
    #                 "--inplace", "/Users/han/Documents/GitHub/Satellite-Project/3DVisual.ipynb"], check=True)
    
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

label1 = tk.Label(root, text="number od radars:", justify="left", anchor="w")
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