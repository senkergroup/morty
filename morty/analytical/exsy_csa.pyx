#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

import math
import numpy as np
cimport numpy as np
import cython
from morty.atomistic import tensor
from itertools import product

ctypedef np.float64_t DTYPE_t

cdef np.ndarray omega_vec(np.ndarray mytensor, np.ndarray z, np.ndarray phi):
    z1 = 1 - z**2
    z2 = np.sqrt(z1)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return (mytensor[0, 0] * z1 * cosphi**2 + mytensor[1, 1] * z1 * sinphi**2 + mytensor[2, 2] * z**2 +
            2 * mytensor[0, 1] * z1 * cosphi * sinphi +
            2 * mytensor[0, 2] * z * z2 * cosphi +
            2 * mytensor[1, 2] * z * z2 * sinphi)

cpdef np.ndarray exsy(axis, np.ndarray tensor1, np.ndarray tensor2, int deg):
    cdef int axis_length = len(axis)
    cdef double start = min(axis)
    cdef double end = max(axis)
    cdef np.ndarray[DTYPE_t, ndim=2] spc_raw = np.zeros((axis_length, axis_length))
    cdef double [:, :] spc = spc_raw
    cdef tuple sampling = np.polynomial.legendre.leggauss(deg)
    cdef np.ndarray[DTYPE_t, ndim=2] z_phi
    cdef np.ndarray[DTYPE_t, ndim=1] sum_weighting
    cdef double calc
    cdef int k
    cdef int l
    cdef double step_size
    z_phi = np.array(list(product(.5 * (sampling[0] + 1), math.pi / 4 * (sampling[0] + 1))))
    z_phi = np.concatenate((z_phi, z_phi + np.array([0, math.pi]), -z_phi + np.array([0, math.pi]),
                            -z_phi + np.array([0, 2*math.pi])))
    sum_weighting = np.prod(list(product(sampling[1], sampling[1])) * 4, axis=1)
    omega1, omega2 = (omega_vec(tensor1, z_phi[:, 0], z_phi[:, 1]),
                      omega_vec(tensor2, z_phi[:, 0], z_phi[:, 1]))

    # this is faster, even without cython, because we handle laaaarge arrays if vectorized!
    step_size = abs(axis[1] - axis[0])
    for k, l in product(np.array(range(axis_length)), range(axis_length)):
        calc = np.sum(sum_weighting * np.clip((1 - np.abs((k * step_size + start) - omega1)) / step_size, 0, 1) *
                      np.clip((1 - np.abs((l * step_size + start) - omega2)) / step_size, 0, 1))
        spc[k, l] += calc
        spc[l, k] += calc
    return spc_raw

cpdef np.ndarray exsy_2(axis, tuple tensors, np.ndarray mytensor2, int deg):
    cdef int axis_length = len(axis)
    cdef double start = min(axis)
    cdef double end = max(axis)
    cdef np.ndarray[DTYPE_t, ndim=2] spc_raw = np.zeros((axis_length, axis_length))
    cdef double [:, :] spc = spc_raw
    cdef tuple sampling = np.polynomial.legendre.leggauss(deg)
    cdef np.ndarray[DTYPE_t, ndim=2] z_phi
    cdef np.ndarray[DTYPE_t, ndim=1] sum_weighting
    cdef double calc
    cdef int k
    cdef int l
    cdef int num_tensors = len(tensors)
    cdef double step_size
    z_phi = np.array(list(product(.5 * (sampling[0] + 1), math.pi / 4 * (sampling[0] + 1))))
    z_phi = np.concatenate((z_phi, z_phi + np.array([0, math.pi]), -z_phi + np.array([0, math.pi]),
                            -z_phi + np.array([0, 2*math.pi])))
    sum_weighting = np.prod(list(product(sampling[1], sampling[1])) * 4, axis=1)
    omegas = [omega_vec(tensor, z_phi[:, 0], z_phi[:, 1]) for tensor in tensors]

    # this is faster, even without cython, because we handle laaaarge arrays if vectorized!
    step_size = abs(axis[1] - axis[0])
    for k, l in product(np.array(range(axis_length)), range(axis_length)):
        calc = sum_weighting
        for i in range(num_tensors):
            calc *= np.clip((1 - np.abs((k * step_size + start) - omegas[i])) / step_size, 0, 1)
        calc = np.sum(calc)
        spc[k, l] += calc
        spc[l, k] += calc
    return spc_raw