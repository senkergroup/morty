"""
A set of mathematical helpers.

The ultimative garbage collector for functions with no home. A collection
of mathematical helpers, which could possibly come in handy in more than one
module or class.
"""

import math
import numpy as np


__all__ = ['axis_rotation_matrix', 'euler_to_spherical',
           'find_nearest_index_by_value', 'spherical_coords',
           'wigner', 'zyz_euler_matrix']


def zyz_euler_matrix(alpha, beta, gamma):
    """
    Euler transform matrix in zyz convention.

    """
    cosa = math.cos(alpha)
    cosb = math.cos(beta)
    cosg = math.cos(gamma)

    sina = math.sin(alpha)
    sinb = math.sin(beta)
    sing = math.sin(gamma)

    return np.array([[cosa * cosb * cosg - sina * sing,
                      -sina * cosg - cosa * cosb * sing,
                      cosa * sinb],
                     [sina * cosb * cosg + cosa * sing,
                      cosa * cosg - sina * cosb * sing,
                      sina * sinb],
                     [-sinb * cosg, sinb * sing, cosb]])


def euler_to_spherical(alpha, beta, gamma):
    """
    Converts Euler angles to spherical coordinates.

    Useful to convert crystal powder data.

    Returns
    -------
    phi, theta : float

    """
    euler_mat = zyz_euler_matrix(alpha, beta, gamma) @ np.array([1, 0, 0])
    return(math.atan(euler_mat[1] / euler_mat[0]), math.acos(euler_mat[2]))


def wigner(alpha, beta, gamma):
    """
    Wigner transformation matrix.

    Parameters
    ----------
    alpha, betta, gamma : float
      Angles in rad.

    Returns
    -------
    rotation_matrix : np.ndarray

    """
    cb = np.cos(beta)
    sb = np.sin(beta)

    a = (1 + cb)**2/4
    b = -(1 + cb)/2*sb
    c = np.sqrt(6) / 4 * sb * sb
    d = -(1 - cb) / 2 * sb
    e = (1 - cb)**2 / 4
    f = (1 + cb) * (2 * cb - 1) / 2
    g = -np.sqrt(3 / 2) * sb * cb
    h = (1 - cb) * (2 * cb + 1) / 2
    k = (3 / 2 * cb * cb - .5)

    D = np.array([[a, b, c, d, e],
                  [-b, f, g, h, d],
                  [c, -g, k, g, c],
                  [-d, h, -g, f, b],
                  [e, -d, c, -b, a]], dtype=np.complex)
    for m in range(2, -3, -1):
        D[2 - m, :] = np.exp(-1j * m * alpha) * D[2 - m, :]
    for n in range(2, -3, -1):
        D[:, 2 - n] = D[:, 2 - n] * np.exp(-1j * n * gamma)
    return D


def axis_rotation_matrix(axis_vector, angle):
    """
    Returns the rotation matrix for a given axis around an angle.

    Parameters
    ----------
    axis_vector : array
        Vector of the axis direction.
    angle : float
        Angle in radians.

    """
    # http://de.wikipedia.org/wiki/Drehmatrix
    cosa = np.cos(angle)
    sina = np.sin(angle)
    return np.array([[np.cos(angle) + axis_vector[0] ** 2 * (1 - cosa),
                      axis_vector[0] * axis_vector[1] * (1 - cosa) -
                      axis_vector[2] * sina,
                      axis_vector[0] * axis_vector[2] * (1 - cosa) +
                      axis_vector[1] * sina],
                     [axis_vector[0] * axis_vector[1] * (1 - cosa) +
                      axis_vector[2] * sina,
                      np.cos(angle) + axis_vector[1] ** 2 * (1 - cosa),
                      axis_vector[1] * axis_vector[2] * (1 - cosa) -
                      axis_vector[0] * sina],
                     [axis_vector[0] * axis_vector[2] * (1 - cosa) -
                      axis_vector[1] * sina,
                      axis_vector[1] * axis_vector[2] * (1 - cosa) +
                      axis_vector[0] * sina,
                      cosa + axis_vector[2] ** 2 * (1 - cosa)]])


def spherical_coords(vector):
    """
    Calculates the sperical coordinates for a given vector.

    Parameters
    ----------
    vector : array_like

    Returns
    -------
    r : float
        Radius of the sphere/length of the vector.
    theta : float
        Theta, the angle between the vector and the c-axis.
    phi : float
        Phi, the angle within the a/b plane.

    """
    radius = np.linalg.norm(vector)
    return radius, math.acos(vector[2] / radius), math.atan2(vector[1],
                                                             vector[0])

def find_nearest_index_by_value(input_array, target_value, nearer=None):
    """
    Find the index corresponding to the best match to ``value`` in an array.

    Parameters
    ----------
    input_array : array
        The array to match within.
    target_value : float
        The value to match with.
    nearer : string, optional
        Find the index for the nearest higher or lower value than the target
        value.<br>
        Accepts 'higher', 'lower' (corresponding to the value, not the index)
        and 'None', defaults to 'None'. None means simply finding the nearest.

    Returns
    -------
    idx : array of int
        The index corresponding to the match.

    """
    if nearer == 'lower':
        idx = np.where(np.abs(input_array - target_value) == np.min(np.abs(
            input_array[np.where(input_array < target_value)] - target_value)))
    elif nearer == 'higher':
        idx = np.where(np.abs(input_array - target_value) == np.min(np.abs(
            input_array[np.where(input_array > target_value)] - target_value)))
    else:
        idx = np.where(np.abs(input_array - target_value) == np.min(np.abs(
            input_array - target_value)))
    return idx[0][0]
