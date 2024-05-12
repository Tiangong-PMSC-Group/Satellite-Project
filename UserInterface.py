import subprocess
import tkinter as tk
from config import config, save_config

def on_button_click():
    # for widget in root.winfo_children():
    #     if widget != button:
    #         widget.grid_remove()

   
    radar_counts = input_radar.get()
    radar_speed = input_speed.get()
    # config['radar']['counts'] = radar_counts
    config['radar']['noise']['rho'] = int(radar_counts)
    save_config(config)

    button.config(text="simulation running...")
    button.config(fg='red')
    root.update()
    # process = subprocess.Popen(["python", "/Users/han/Documents/GitHub/Satellite-Project/Visualization.py"])
    # subprocess.Popen(["python", "/Users/han/Documents/GitHub/Satellite-Project/Visualization.py"])
    # subprocess.run(["python", "/Users/han/Documents/GitHub/Satellite-Project/Visualization.py"], check=True)
    subprocess.run(["python", "Visualization.py"], check=True)
    subprocess.run(["python", "try.py"], check=True)
    # subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute",
    #                 "--inplace", "/Users/han/Documents/GitHub/Satellite-Project/3DVisual.ipynb"], check=True)
    
    button.config(text="finished, please check the display window.")
    root.update()

    
    root.after(3000, root.destroy)


root = tk.Tk()

# root.geometry("400x300")  

# root.resizable(width=False, height=False)

i = 0
label_radar = tk.Label(root, text="radar ", justify="left")
label_radar.grid(row=i, column=0, padx=10, pady=(0, 0))  


i=i+1

label1 = tk.Label(root, text="count:")
label1.grid(row=i, column=0, padx=10, pady=(0, 0))  
input_radar = tk.Entry(root)
# input_radar.insert(0, config['radar']['counts'])
input_radar.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))  

i = i + 1
label_radar_hint = tk.Label(root, text="hint ")
label_radar_hint.grid(row=i, column=0, columnspan=2, padx=10, pady=(0, 0))


i=i+1

label2 = tk.Label(root, text="radar mode:")
label2.grid(row=i, column=0, padx=10, pady=(0, 0))  
input_speed = tk.Entry(root)
input_speed.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))  

i=i+1
label3 = tk.Label(root, text="radar speed:")
label3.grid(row=i, column=0, padx=10, pady=(0, 0))  
input_speed1 = tk.Entry(root)
input_speed1.grid(row=i, column=1, padx=(0, 10), pady=(0, 0))  

i=i+1

button = tk.Button(root, text="submit", command=on_button_click)
button.grid(row=i, column=0, columnspan=2, pady=5)


root.mainloop()