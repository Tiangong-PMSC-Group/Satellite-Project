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
