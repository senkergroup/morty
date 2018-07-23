"""
Provides tools to calculate CSA lineshapes with analytical equations. :cite:`Macho2001`

"""

import math
import itertools
import numpy as np
from numpy.polynomial import legendre
import scipy.linalg
from scipy.ndimage.filters import gaussian_filter1d
from ..util import wigner
from ..util import euler_to_spherical


__all__ = ['zcw', 'intermediate_csa_lineshape', 'intermediate_csa_lineshape_legacy', 'csa', 'csa_legacy', 'exsy']


def zcw(num_angles, spherical=True):
    """
    Calculate powder angles and their weighting using the zcw scheme.

    Parameters
    ----------
    num_angles : int
        Specifies the number of the calculated angles. The number of angles
        returned will be the next Fibonacci number that is equal or higher than
        the provided value.
    spherical : bool, optional
        If :const:`True` (default), the result will be spherical angles, otherwise it's
        Euler angles.

    Returns
    -------
    alpha, beta : float
        Euler or spherical angles in radians.
    weighting : float
        Weighting.

    """

    fib1, fib2 = 1, 1
    while fib1 + fib2 < num_angles:
        fib1, fib2 = fib2, fib1 + fib2
    n = fib1 + fib2
    pin = math.pi / n
    i = np.linspace(0, n - 1, n)
    angles = np.array([(i + .5) * pin, i * fib1 % n * 2 * pin,
                       np.sin((i + .5) * pin)]).T
    if spherical is True:
        for angle in range(len(angles)):
            angles[angle, 0:2] = euler_to_spherical(
                angles[angle, 0], angles[angle, 1], 0)
    return angles


def intermediate_csa_lineshape_legacy(dwell, fid_size, rate_mat, sites,
                               powder_angles=1000, lb=.1e3):
    """
    Calculate FID and spectrum for CSA lineshapes with jump motion.

    Parameters
    ----------
    dwell : float
        In s.
    fid_size : int
        Number of FID points.
    rate_mat : array_like
        The rate matrix including the jump rate.
    sites : tuple
        A tuple of sites, each containing (anisotropy, asymmetry, alpha, beta,
        gamma, probability). Alpha, beta and gamma are the Euler angles
        describing the relative orientation of the jump sites.
    powder_angles : int
        Number of zcw powder angles used. Something between 500 and 20.000 is
        reasonable, depending on linebroadening, anisotropy and asymmetry.

    Returns
    -------
    fid : array
        The complex FID.
    fft_freq : np.ndarray
        The frequency axis.
    fft : np.ndarray
        The spectrum.

    Examples
    --------
    Two-site jump with an anisotropy of 3 kHz, asymmetry of 0.5, a jump rate
    of 1 kHz and a jump angle of 90°. ::

        csa_motion_fid(100e-6, 128, np.array([[-1, 1], [1, -1]]) * 1e3,
                       ((3e3, 0.5, 0, 0, 0, 1),
                        (3e3, 0.5, 0, math.pi/2, 0, 1)),
                       powder_angles=10000)

    """
    sites = np.array(sites)
    angles = zcw(powder_angles, spherical=False)
    lb = np.diag([lb] * len(sites))
    omegas = np.zeros((len(sites), len(sites)))
    time_axis = np.arange(0, fid_size * dwell, dwell)
    fid = np.zeros(fid_size, dtype=np.complex)
    # Calculate the wigner matrix for the relative site angles.
    wigner_site = [wigner(site[2], site[3], site[4]) for site in sites]
    mag_zero = np.array([x for x in sites[:, 5]],
                        dtype=np.complex)
    # Now iterate over the powder angles.
    for angle in angles:
        for i, site in enumerate(sites):
            # Values in rad/sec
            delta = site[0] * 2 * math.pi
            eta = delta * site[1] / 2.4494897427831779
            omegas[i, i] = (np.array([eta, 0, delta, 0, eta]) @
                            wigner_site[i] @ wigner(angle[1], angle[0], 0))[2]
        prop = scipy.linalg.expm((1j * omegas + rate_mat - lb) * dwell)
        mag_z = mag_zero.copy()
        fid[0] += mag_zero @ mag_z * angle[2]
        for i in range(1, fid_size - 1):
            mag_z = prop @ mag_z
            fid[i] += mag_zero @ mag_z * angle[2]
    fid[0] = .5 * (fid[0] + fid[-1])
    fft = np.fft.fft(fid)
    fft_freq = np.fft.fftfreq(len(time_axis), time_axis[1] - time_axis[0])
    return fid, np.sort(fft_freq), np.array([x for (y, x) in sorted(zip(fft_freq, fft))])


def _csa_calculate(cos_theta, sin_theta_cos_2_phi, aniso, asym, omega, lb, gb):
    """
    Calculates the intensity of one point of a CSA lineshape at given angles.

    You usually do not use this. You need to integrate over phi and theta. See
    :class:`morty.analytical.csa`.

    Parameters
    ----------
    cos_theta : array_like
        3 * cos(Theta)**2 - 1
    sin_theta_cos_2_phi : array_like
        sin(Theta) * cos(2 * Phi)
    aniso : float
    asym : float
    omega : array
    lb : float

    """
    fromiso = .5 * aniso * (cos_theta - asym * sin_theta_cos_2_phi) - omega
    return (np.exp(-0.69314718055994529 * 4 * (fromiso / gb) ** 2) +
            (1 / (1 + (fromiso / lb * 2) ** 2)))

def _omega_vec(mytensor, z, phi):
    z1 = 1 - z**2
    z2 = np.sqrt(z1)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return (mytensor[0, 0] * z1 * cosphi**2 + mytensor[1, 1] * z1 * sinphi**2 + mytensor[2, 2] * z**2 +
            2 * mytensor[0, 1] * z1 * cosphi * sinphi +
            2 * mytensor[0, 2] * z * z2 * cosphi +
            2 * mytensor[1, 2] * z * z2 * sinphi)

def intermediate_csa_lineshape(dwell, fid_size, rate_mat, sites,
                               deg=30, lb=.1e3):
    """
    Calculate FID and spectrum for CSA lineshapes with jump motion.

    Parameters
    ----------
    dwell : float
        In s.
    fid_size : int
        Number of FID points.
    rate_mat : array_like
        The rate matrix including the jump rate.
    sites : tuple
        A tuple of sites, each containing (anisotropy, asymmetry, alpha, beta,
        gamma, probability). Alpha, beta and gamma are the Euler angles
        describing the relative orientation of the jump sites.
    deg : int
        Degree of Legendre polynomial used for the Gaussian quadrature
        integration.

    Returns
    -------
    fid : array
        The complex FID.
    fft_freq : np.ndarray
        The frequency axis.
    fft : np.ndarray
        The spectrum (complex).

    Examples
    --------
    Two-site jump with an anisotropy of 3 kHz, asymmetry of 0.5, a jump rate
    of 1 kHz and a jump angle of 90°. ::

        csa_motion_fid(100e-6, 128, np.array([[-1, 1], [1, -1]]) * 1e3,
                       ((tensor, 1),
                        (tensor2, 1)))

    """
    sampling = legendre.leggauss(deg)
    z_phi = np.array(list(itertools.product(.5 * (sampling[0] + 1), math.pi / 4 * (sampling[0] + 1))))
    sum_weighting = np.prod(list(itertools.product(sampling[1], sampling[1])), axis=1)
    lb = np.diag([lb] * len(sites))
    omegas = np.zeros((len(sites), len(sites)), dtype=np.complex)
    time_axis = np.arange(0, fid_size * dwell, dwell)
    fid = np.zeros(fid_size, dtype=np.complex)
    sites = np.array(sites)
    mag_zero = np.array([x for x in sites[:, 1]],
                        dtype=np.complex)

    # Now iterate over the powder angles.
    for i in range(len(z_phi)):
        for j, site in enumerate(sites):
            omegas[j, j] = _omega_vec(site[0].tensor, z_phi[i, 0], z_phi[i, 1]) * 2 * math.pi * 1j
        prop = scipy.linalg.expm((omegas + rate_mat - lb) * dwell)
        mag_z = mag_zero.copy()
        fid[0] += mag_zero @ mag_z * sum_weighting[i]
        for j in range(1, fid_size - 1):
            mag_z = prop @ mag_z
            fid[j] += mag_zero @ mag_z * sum_weighting[i]
    fid[0] = .5 * (fid[0] + fid[-1])
    fft = np.fft.fft(fid)
    fft_freq = np.fft.fftfreq(len(time_axis), time_axis[1] - time_axis[0])
    return fid, np.sort(fft_freq), np.array([x for (y, x) in sorted(zip(fft_freq, fft))])

def exsy(axis, tensor1, tensor2, deg):
    start, end = min(axis), max(axis)
    step_size = abs(axis[1] - axis[0])
    mysize = len(axis)
    spc = np.zeros((mysize, mysize))
    sampling = np.polynomial.legendre.leggauss(deg)
    z_phi = np.array(list(itertools.product(.5 * (sampling[0] + 1), math.pi / 4 * (sampling[0] + 1))))
    z_phi = np.concatenate((z_phi, z_phi + np.array([0, math.pi]), -z_phi + np.array([0, math.pi]),
                            -z_phi + np.array([0, 2*math.pi])))
    sum_weighting = np.prod(list(itertools.product(sampling[1], sampling[1])) * 4, axis=1)
    omega1, omega2 = (omega_vec(tensor1, z_phi[:, 0], z_phi[:, 1]),
                      omega_vec(tensor2, z_phi[:, 0], z_phi[:, 1]))

    step_size = (end - start) / (mysize - 1)
    # this is faster, even without cython, because we handle laaaarge arrays if vectorized!
    for k, l in itertools.product(range(mysize), range(mysize)):
        calc = np.sum(sum_weighting * np.clip((1 - np.abs((k * step_size + start) - omega1)) / step_size, 0, 1) *
                      np.clip((1 - np.abs((l * step_size + start) - omega2)) / step_size, 0, 1))
        spc[k, l] += calc
        spc[l, k] += calc
    return spc

def csa(omegas, aniso, asym, iso=0, lb=1, gb=1, deg=100, scaling=1):
    """
    Calculates a CSA lineshape.

    Parameters
    ----------
    omegas : array_like
        Values in ppm/hertz where the amplitude is calculated.
    aniso : float
        Anisotropy in ppm/hertz.
    asym : float
        Asymmetry.
    lb : float
        Gaussian linebroadening in ppm/hertz.
    deg : int
        Degree of Legendre polynomial to use for the Gaussian quadrature
        integration. Depending on linebroadening and asymmetry, values
        between 50 and 150 are reasonable.
    scaling : float
        Scale the result.

    """
    spc = np.zeros(len(omegas))
    sampling = legendre.leggauss(deg)
    z_phi = np.array(list(itertools.product(.5 * (sampling[0] + 1), math.pi / 4 * (sampling[0] + 1))))
    #z_phi = np.concatenate((z_phi, z_phi + np.array([0, math.pi]), -z_phi + np.array([0, math.pi]), -z_phi + np.array([0, 2*math.pi])))
    sum_weighting = np.prod(list(itertools.product(sampling[1], sampling[1])), axis=1)
    #sum_weighting = np.prod(list(itertools.product(sampling[1], sampling[1])) * 4, axis=1)

    z1 = 1 - z_phi[:, 0]**2
    z2 = np.sqrt(z1)
    cosphi = np.cos(z_phi[:, 1])
    sinphi = np.sin(z_phi[:, 1])
    omega_calc = (-.5 * aniso * (1 + asym) * z1 * cosphi**2 + -.5 * aniso * (1 - asym) * z1 * sinphi**2 + aniso * z_phi[:, 0]**2)

    step_size = abs(omegas[0] - omegas[1])
    for i, omega in enumerate(omegas):
        spc[i] += np.sum(sum_weighting * np.clip(1 - np.abs(omega - omega_calc - iso) / step_size, 0, 1))
    spc = gaussian_filter1d(spc, gb / step_size)
    lorentzian = lambda x, x0, fwhm: 1 / (1 + ((x - x0) / fwhm * 2)**2)
    lorentz_filtered = np.zeros(len(spc))
    for i, val in enumerate(spc):
        lorentz_filtered += val * lorentzian(omegas, omegas[i], lb)
    return lorentz_filtered / max(lorentz_filtered) * scaling


def csa_legacy(omegas, aniso, asym, iso=0, lb=1, gb=1, powder_angles=1000, scaling=1):
    """
    Calculates a CSA lineshape.

    Parameters
    ----------
    omegas : array_like
        Values in ppm/hertz where the amplitude is calculated.
    aniso : float
        Anisotropy in ppm/hertz.
    asym : float
        Asymmetry.
    lb : float
        Gaussian linebroadening in ppm/hertz.
    powder_angles : int or array_like
        Number of zcw powder angles to be used for the calculation or an array
        of powder angles, which can be created by :class:`morty.analytical.zcw()`.
        Try experimenting: smaller linebroadening and a higher
        asymmetry * anisotropy requires larger values. A good resultion
        should be achievable with values between 100 (eta = 0) and 20.000.
        Beware: not the exact number will be used, but the next higher
        Fibonacci number.
        On the fly calculation takes time. You can also supply the powder
        angles yourself.
    scaling : float
        Scale the result.

    Examples
    --------
    To speed up the calculation, for example when used by an optimizer, use ::

        powder_angles = zcw(754)
        csa(np.linspace(-50, 50), 50, 0.2, powder_angles=powder_angles)

    This calculates from -50 to 50 (ppm or Hz - just use the same for the axis as
    for the anisotropy) with an anisotropy of 50 and a asymmetry of 0.2.

    """
    if isinstance(powder_angles, int):
        angles = zcw(powder_angles)
    else:
        angles = powder_angles

    # np.newaxis creates a 2D vector, which can be transposed to achieve the
    # vectorization of the function:
    # we want to calculate the sum over all angles for each given omega.
    cos_theta = 3 * np.cos(angles[:, 0]) ** 2 - 1
    sin_theta_cos_2_phi = np.sin(angles[:, 0])**2 * np.cos(2 * angles[:, 1])
    spec = np.sum(_csa_calculate(cos_theta, sin_theta_cos_2_phi, aniso,
                                 asym, omegas[np.newaxis].T - iso, lb, gb) *
                  angles[:, 2], axis=1)
    return spec / np.max(spec) * scaling
