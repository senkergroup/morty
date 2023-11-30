"""
Handles Pseudo Voigt Lineshapes, which are mainly used for deconvolution of
e.g. :class:`morty.analytical.Spectrum1D.integrate_deconvoluted()`

"""

import numpy as np


__all__ = ['pseudovoigt', 'pseudovoigt_integral', 'pseudovoigt_sum']


def pseudovoigt(x_axis, iso, sigma, gamma, intensity, eta):
    """
    Amplitude-normalised Pseudo Voigt profile.

    This is build of a Gaussian :math:`G` and a Lorentzian :math:`F`:

    .. math::
      V_p(x) = eta × G + (1-eta) × L

    Parameters
    ----------
    x_axis : array_like
        x-axis to calculate the PsV profile for.
    iso : float
        Center of the PV shape, corresponding to the isotropic shift of
        the respective signal.
    sigma : float
        FWHM of gaussian line.
    gamma : float, optional
        FWHM of the Lorentzian line. If you wish to plot a PsV with the same
        fwhm for Gaussian and Lorentzian shape, you can just omit gamma.
    intensity : float
        Intensity, i.e. amplitude, of the whole profile.
    eta : float
        Mixing parameter for Gaussian and Lorentzian shape. (0-1)

    Returns
    -------
    pseudovoigt : np.ndarray
        Pseudo Voigt Profile for given params of same length as s_axis.

    """
    if gamma is None:
        gamma = sigma

    return (intensity * (eta * np.exp(
        -0.69314718055994529 * 4 * ((x_axis - iso) / sigma) ** 2) +
                         (1 - eta) * (1 / (1 + 4 * ((x_axis - iso) /
                                                    gamma) ** 2))))


def pseudovoigt_integral(sigma, gamma, intensity, eta):
    """
    Integral of the amplitude-normalised PsV profile.

    This is build of a Gaussian :math:`G` and a Lorentzian :math:`F`:

    .. math::
      V_p(x) = eta × G + (1-eta) × L

    Parameters
    ----------
    sigma, gamma : float
        FWHM of the gaussian and lorenzian part of the function.
    intensity : float
        Amplitude of the function.
    eta : float
        Mixing parameter for Gaussian and Lorentzian shape. (0-1)

    Returns
    -------
    integral : float
        The integral for the amplitude-normalised PsV profile.

    """
    # sqrt(pi/ln(2)) = 2.1289340388624525
    return (intensity * (eta * (sigma / 2) * 2.1289340388624525 +
                         (1 - eta) * gamma * (np.pi / 2)))


def pseudovoigt_sum(x_axis, iso, sigma, gamma, intensity, eta):
    """
    Sums up multiple Pseudo Voigt functions.

    This is build of a Gaussian :math:`G` and a Lorentzian :math:`F`:

    .. math::
      V_p(x) = eta × G + (1-eta) × L

    Parameters
    ----------
    x_axis : array_like
        Points at which the PV functions are evaluated.
    iso : array_like
        Array of isotropic shifts.
    sigma : array_like
        Array of FWHM of gaussian lines.
    gamma : array_like
        Array of FWHM of the Lorentzian lines.
    intensity : array_like
        Array of amplitudes.
    eta : array
        Mixing parameter for Gaussian and Lorentzian shape. (0-1)
    
    Returns
    -------
    sum : np.ndarray
        Sum of all Pseudo Voigt functions.

    """
    return np.sum((pseudovoigt(x_axis, iso[i], sigma[i], gamma[i],
                               intensity[i], eta[i])
                   for i in range(len(iso))), axis=0)
