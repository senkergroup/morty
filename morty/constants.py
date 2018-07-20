"""
General constants.

Please note that most physical constants are available via the scipy.constants
package.

.. rubric:: Constants

+-----------------------------------+-----------------+------+
|     Description                   |  in units of    | cit. |
+===================================+=================+======+
|      GYROMAGNETIC_RATIOS          |   rad/(T*s)     |   1  |
+-----------------------------------+-----------------+------+
|      QUADRUPOLE_MOMENT            |   m**2          |   1  |
+-----------------------------------+-----------------+------+
|      NMR_MASSNUMBERS              |   u             |   1  |
+-----------------------------------+-----------------+------+
|      NUC_CHARGES                  |   e             |   1  |
+-----------------------------------+-----------------+------+
|      VDW_RADII                    |   m             |   2  |
+-----------------------------------+-----------------+------+

1. Table taken from Almanac 2006, Bruker, ``NMR Properties of Selected
   Isotopes``
2. de.wikipedia.org/wiki/Van-der-Waals-Radius

"""

GYROMAGNETIC_RATIOS = {
    '1H': 26.75222127e7, '6Li': 3.9371273e7,
    '7Li': 10.3977047e7, '13C': 6.728286e7,
    '14N': 1.93377981e7, '15N': -2.712618911e7,
    '19F': 25.162333e7, '23Na': 7.0808515e7,
    '27Al': 6.97627808e7, '29Si': -5.319031e7,
    '31P': 10.83941e7, '129Xe': -7.452103e7,
    '25Mg': -1.638843e7}

QUADRUPOLE_MOMENT = {
    '2H': 0.286e-30, '7Li': -4.01e-30, '14N': 2.04e-30, '17O': -2.558e-30,
    '23Na': 10.4e-30, '25Mg': 19.94e-30, '27Al': 14.66e-30}

NMR_MASSNUMBERS = {
    'Al': 27, 'C': 13, 'H': 1, 'Li': 7, 'Mg': 25,
    'N': 15, 'O': 17, 'F': 19, 'Na': 23, 'Si': 29, 'P': 31,
    'Xe': 129}

NUC_CHARGES = {
    'H': 1, 'Li': 3, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Al': 13,
    'Si': 14, 'S': 16, 'Cl': 17, 'Mg': 12, 'Xe': 54, 'P': 15}

VDW_RADII = {
    'H': 1.1e-10, 'He': 1.4e-10, 'Li': 1.82e-10, 'Be': 1.53e-10,
    'B': 1.92e-10, 'C': 1.70e-10, 'N': 1.55e-10, 'O': 1.52e-10,
    'F': 1.42e-10, 'Ne': 1.54e-10, 'Na': 2.27e-10, 'Mg': 1.73e-10,
    'Al': 1.84e-10, 'Si': 2.10e-10, 'P': 1.80e-10, 'S': 1.80e-10,
    'Cl': 1.75e-10, 'Ar': 1.88e-10, 'Ga': 1.87e-10, 'In': 2.20e-10,
    'Xe': 2.16e-10, 'V': 2.05e-10}

ATOM_TYPES = {
    'H': 'hydrogen', 'He': 'helium', 'Li': 'lithium', 'Be': 'beryllium',
    'B': 'boron', 'C': 'carbon', 'N': 'nitrogen', 'O': 'oxygen',
    'F': 'fluorine', 'Ne': 'neon', 'Na': 'sodium', 'Mg': 'magnesium',
    'Al': 'aluminum', 'Si': 'silicon', 'P': 'phosphorus', 'S': 'sulfur',
    'Cl': 'chlorine', 'Ar': 'argon', 'Xe': 'xenon'}
