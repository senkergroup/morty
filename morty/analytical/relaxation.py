"""
Functions to describe NMR relaxation.

"""

import numpy as np


__all__ = ['t1_relaxation']


def t1_relaxation(t1, time, measurement='satrecovery'):
    """
    Returns the time-dependant signal amplitude for T1 relaxation.

    Calculates the amplitudes for a saturation recovery or inversion recovery
    experiment. The amplitude is normalized.

    Parameters
    ----------
    t1 : float
        T1 in s (or the same unit as `t`).
    time : float or array_like
        Time at which the amplitude will be calculated.
    measurement : 'satrecovery' or 'invrecovery'
        Simulate amplitude for saturation recovery or inversion recovery.

    Returns
    -------
    intensity : float or np.ndarray
        Relative intensity, as float or array, depending on the supplied `t`.

    """
    if measurement == 'satrecovery':
        factr = 1
    else:
        factr = 2
    return 1 - factr * np.exp(-time / t1)
