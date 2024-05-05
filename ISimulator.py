'''
  For the third step, I think using interface is a good idea.When we want to debug radars all debug predictors,
  simulate trajectories every time will waste time.
  So we can have two implementations of the same Interface(ISimulator),
  One is called FakeSimulator, which return exsiting data from storage,
   another one is the real simulator which update dynamics at each time step .
   By using interface,we can switch these two mode very quickly
'''
from abc import abstractmethod, ABCMeta
import pickle
from SatelliteState import SatelliteState
from config import config

class ISimulator(metaclass=ABCMeta):

    def __init__(self):
        self.G = config['sim_config']['gravitational_constant']
        self.M = config['earth']['mass']
        self.dt = config['sim_config']['dt']['main_dt']

    '''
    For each timestep, simulate the position and return it.
    '''

    @abstractmethod
    def simulate(self, current_time) -> SatelliteState:
        raise NotImplementedError


''' '''


class FakeSimulator(ISimulator):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.simulation_data = self.load_from_file()

    def load_from_file(self):
        with open(self.filename, 'rb') as file:
            data = pickle.load(file)
            print("read data finished")
            return data

    def simulate(self, current_time) -> SatelliteState:
        if current_time < len(self.simulation_data):
            return self.simulation_data[current_time]
        else:
            raise Exception("No more data available.")


class RealSimulator(ISimulator):
    """Any class can be a child of the ISimulator ,just need to extend this interface and 
    implement the simulate method"""

    def __init__(self):
        super().__init__()

    def simulate(self) -> SatelliteState:
        """The real dynamics which can generate the real positions of satellites"""
        raise NotImplementedError
