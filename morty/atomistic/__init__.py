"""
This package is dedicated to everything related to meddling with Cells, Atoms
and the various cool things one can do with them.
This includes simple geometries and transformation of them, as well as handling
properties like charges, chemical shifts and more.
The structures build on each other, so a *Cell* contains a list of *Atom* s,
adding properties necessary for a cell, like unit cell constants, or the
ability to rotate groups of atoms.

"""

from .cell import *
from .cellmodeller import *
from .tensor import *
from .spinsystem import *
from .atom import *
from .trajectory import *

__all__ = ['Atom', 'Cell', 'CellModeller', 'Spinsystem',
           'CSATensor', 'DipoleTensor', 'EFGTensor', 'Trajectory']
