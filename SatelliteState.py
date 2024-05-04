from dataclasses import dataclass
import numpy as np

''' 
A dataclass which now is combination of time and position, and if the following steps need more data from the simulator,
add a field here is very convenient to let data pass through classes
'''
@dataclass
class SatelliteState:
    velocity: np.ndarray
    pos: np.ndarray
    current_time: int

