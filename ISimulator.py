'''
  For the third step, I think using interface is a good idea.When we want to debug radars all debug predictors,
  simulate trajectories every time will waste time.
  So we can have two implementations of the same Interface(ISimulator),
  One is called FakeSimulator, which return exsiting data from storage,
   another one is the real simulator which update dynamics at each time step .
   By using interface,we can switch these two mode very quickly
'''
from abc import abstractmethod, ABCMeta

from SatelliteState import SatelliteState


class ISimulator(metaclass=ABCMeta):

    def __init__(self):
        pass

    '''
    For each timestep, simulate the position and return it.
    '''

    @abstractmethod
    def simulate(self) -> SatelliteState:
        raise NotImplementedError


''' '''


class FakeSimulator(ISimulator):

    def __init__(self):
        super().__init__()

    def simulate(self) -> SatelliteState:
        """TODO read positions from file"""
        raise NotImplementedError


class RealSimulator(ISimulator):
    """Any class can be a child of the ISimulator ,just need to extend this interface and 
    implement the simulate method"""

    def __init__(self):
        super().__init__()

    def simulate(self) -> SatelliteState:
        """The real dynamics which can generate the real positions of satellites"""
        raise NotImplementedError
