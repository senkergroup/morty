"""
Modules to handle NMR spectra and methods to analyze them.

Provides tools to read-in TOPSPIN processed data folders, process them and
perform analytical evaluation, like lineshape fitting and simulation of buildup
curves.

"""

from ._csa import *
from .ddevolution import *
from .lineshapes import *
from .relaxation import *
from .spectra import *
try:
	from .exsy_csa import exsy_csa
except ImportError:
	pass
