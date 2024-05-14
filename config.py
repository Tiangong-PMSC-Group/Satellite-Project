import json
import os


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

'''
Use absolute path so that Classes in other directory can load config too
'''
path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(path)
path_in = os.path.join(os.path.dirname(__file__), 'config_in.json')
config_in = load_config(path_in)

def save_config(data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

# config['radar']['noise']['rho'] = 99
# save_config(config)

