import json
import os


def load_config(file_path):
    """
       Load a JSON configuration file.

       Args:
           file_path (str): The path to the JSON configuration file.

       Returns:
           dict: The loaded configuration as a dictionary.
       """
    with open(file_path, 'r') as file:
        return json.load(file)

'''
Use absolute path so that Classes in other directory can load config too
'''
path = os.path.join(os.path.dirname(__file__), 'configs/config.json')
config = load_config(path)
path_in = os.path.join(os.path.dirname(__file__), 'configs/config_in.json')
config_in = load_config(path_in)

def save_config(data):
    """
       Save a dictionary to a JSON configuration file.

       Args:
           data (dict): The data to save to the JSON file.
       """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)



