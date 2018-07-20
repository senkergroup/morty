"""
Toolkit to set up automated calculations from structures

"""

from .simpson import *
from .dft import *

__all__ = ['SimpsonCaller', 'SpinsystemCreator', 'DFTCaller']