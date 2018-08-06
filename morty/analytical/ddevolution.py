"""
Analytical functions to simulate the evolution of the signal amplitude for
various pulse sequences, which may recouple the DD coupling and allow for the
extraction of distance informations by evaluation of the change of the signal
amplitude over time.

.. rubric:: Constants

==========     ==============================================================
KAPPA_SR26     Scaling factor for analytical evaluation of the SR26 pulse
               sequence.
KAPPA_PC7      Scaling factor for analytical evaluation of the POST-C7 pulse
               sequence.

==========     ==============================================================

"""

import math
import scipy.constants
import scipy.special
import numpy as np
from . import zcw


KAPPA_SR26 = 0.171
KAPPA_PC7 = 0.232

__all__ = ['KAPPA_SR26', 'KAPPA_PC7', 'dd_coupling_constant', 'dist_from_dd_coupling_constant',
           'dq_buildup_sym', 'dq_buildup_ct', 'dq_gamma_buildup_sym', 'dq_gamma_buildup_ct',
           'redor_sum', 'sq_spinecho']


def dd_coupling_constant(distance, gamma1, gamma2=None):
    """
    Calculate the dipolar coupling constant.

    Calculate the dipolar coupling constant in rad/s. The dipolar coupling constant is defined as:

    .. math::
        d_{IS} = -\\frac{\\gamma_I \\gamma_S \\hbar}{2\\pi r^3_{IS}}
            \\frac{\\mu_0}{4\\pi}

    Mind the negative sign, as opposed to other literature.

    Parameters
    ----------
    distance : float
        Distance between the nuclei in m.
    gamma1 : float
        Gyromagnetic ratio (in rad/(T*s)) of the first nucleus.
        This is usually also a `property` of an :class:`morty.atomistic.Atom`.
    gamma2 : float, optional
        Gyromagnetic ratio of the second nucleus. Defaults to :const:`None`, in which
        case both atoms are assumed to have the same gyromagnetic ratio.

    Returns
    -------
    b : float
        Coupling constant in rad/s.

    """

    if gamma2 is None:
        gamma2 = gamma1
    return (- scipy.constants.mu_0 / 4 / scipy.constants.pi * gamma1 *
            gamma2 * scipy.constants.hbar / distance ** 3)


def dist_from_dd_coupling_constant(ddcc, gamma1, gamma2):
    """
    Calculate the distance from a dipolar coupling constant.

    The distance from a dipolar coupling constant is defined as:

    .. math::
        r_{IS} = \\left(\\frac{\\gamma_I \\gamma_S \\hbar}{d_{IS}}
            \\frac{\\mu_0}{4\\pi}\\right)^{\\frac{1}{3}}

    Parameters
    ----------
    r : float
        Distance between the nuclei in m.
    gamma1 : float
        Gyromagnetic ratio (in rad/(T*s)) of the first nucleus.
        This is usually also a `property` of an :class:`morty.atomistic.Atom`.
    gamma2 : float, optional
        Gyromagnetic ratio of the second nucleus. Defaults to :const:`None`, in which
        case both atoms are assumed to have the same gyromagnetic ratio.

    """
    if gamma2 is None:
        gamma2 = gamma1
    return math.fabs(scipy.constants.mu_0 / 4 / scipy.constants.pi *
                     gamma1 * gamma2 * scipy.constants.hbar / ddcc) ** (1/3)


def dq_buildup_sym(time, coupling_constant, kappa):
    """
    Calculation of symmetric buildup of DQ efficiency.

    Calculation of symmetric buildup of DQ efficiency for non gamma-encoded
    sequences, e.g. SR26.

    Parameters
    ----------
    time : float or array_like
        Time (in s) of the excitation of the DQ coherence.
    coupling_constant : float
        Dipolar coupling constant of the nuclei in rad/s.
    kappa : float
        Scaling factor of the pulse sequence. Some constants are defined in
        this module, e.g. `KAPPA_SR26`.

    Returns
    -------
     dq_eff : float or np.ndarray
        Double quantum efficiency. This will be a numpy array or a float,
        depending on the type supplied for `t`.

    """

    ret = (0.5 - ((np.sqrt(2) * np.pi) / 8) *
           scipy.special.jn(0.25, ((3 / 2) * kappa * coupling_constant * time)
                            + 0j) *
           scipy.special.jn(-0.25, ((3 / 2) * kappa * coupling_constant * time) +
                            0j))

    if isinstance(ret, np.ndarray):
        return np.real(np.array([0 if np.isnan(x) else x for x in ret]))
    return np.real(0 if np.isnan(ret) else ret)


def dq_buildup_ct(texc, ttot, coupling_constant, kappa):
    """
    Calculation of a ct buildup curve for non gamma-encoded DQ sequences, e.g.
    SR26.

    Parameters
    ----------
    texc : float or array_like
        Time (in s) of the excitation of the DQ coherence.
    ttot : float
        Time (in s) of the total excitation/reconversion.
    coupling_constant : float
        Dipolar coupling constant of the nuclei in rad/s.
    kappa : float
        Scaling factor of the pulse sequence. Some constants are defined in
        this module, e.g. `KAPPA_SR26`.

    Returns
    -------
     dq_eff : float or np.ndarray
        Double quantum efficiency. This will be a numpy array or a float,
        depending on the type supplied for `texc`.

    """
    # convert to -ttot/2 <= texc <= ttot/2 style:
    texc -= ttot * .5

    bessel_arg1 = (3 / 2) * kappa * coupling_constant * texc
    bessel_arg2 = (3 / 4) * kappa * coupling_constant * ttot
    ret = ((np.sqrt(2) * np.pi) /
           8) * (scipy.special.jn(0.25, bessel_arg1 + 0j) *
                 scipy.special.jn(-0.25, bessel_arg1 + 0j) -
                 scipy.special.jn(0.25, bessel_arg2 + 0j) *
                 scipy.special.jn(-0.25, bessel_arg2 + 0j))

    if isinstance(ret, np.ndarray):
        return np.real(np.array([0 if np.isnan(x) else x for x in ret]))
    return np.real(0 if np.isnan(ret) else ret)


def dq_gamma_buildup_sym(time, coupling_constant, kappa):
    """
    Calculation of symmetric buildup of DQ efficiency for gamma-encoded
    sequences, e.g. POST-C7.

    Parameters
    ----------
    time : float or array_like
        Time (in s) of the excitation of the DQ coherence.
    coupling_constant : float
        Dipolar coupling constant of the nuclei in rad/s.
    kappa
        Scaling factor of the pulse sequence. Some constants are defined in
        this module, e.g. `KAPPA_SR26`.

    Returns
    -------
     dq_eff : float or np.ndarray
        Double quantum efficiency. This will be a numpy array or a float,
        depending on the type supplied for `t`.

    """
    theta = kappa * coupling_constant * time
    x = (2 * theta / np.pi) ** (.5)

    fresin, frecos = scipy.special.fresnel(x * np.sqrt(2))
    ret = .5 - 1 / (x * np.sqrt(8)) * (frecos * np.cos(2 * theta) +
                                       fresin * np.sin(2 * theta))

    if isinstance(ret, np.ndarray):
        return np.real(np.array([0 if np.isnan(x) else x for x in ret]))
    return np.real(0 if np.isnan(ret) else ret)


def dq_gamma_buildup_ct(texc, ttot, coupling, kappa):
    """
    Calculation of a ct buildup curve for gamma-encoded DQ sequences, e.g.
    POST-C7.

    Parameters
    ----------
    texc : float or array_like
        Time (in s) of the excitation of the DQ coherence.
    ttot : float
        Time (in s) of the total excitation/reconversion.
    coupling : float
        Dipolar coupling constant of the nuclei in rad/s.
    kappa : float
        Scaling factor of the pulse sequence. Some constants are defined in
        this module, e.g. `KAPPA_SR26`.

    Returns
    -------
     dq_eff : float or np.ndarray
        Double quantum efficiency. This will be a numpy array or a float,
        depending on the type supplied for `texc`.

    """
    if isinstance(texc, np.ndarray):
        texc = np.array([x + 1e-10 if x == .5 * ttot else x for x in texc])
    else:
        texc = texc + 1e-10 if texc == .5 * ttot else texc

    theta_exc = (kappa * coupling * texc) + 0j
    theta_rec = (kappa * coupling * (ttot - texc)) + 0j

    theta_delta = theta_exc - theta_rec
    theta_sum = theta_exc + theta_rec
    x_delta = (2 * theta_delta / np.pi) ** (1 / 2)
    x_sum = (2 * theta_sum / np.pi) ** (1 / 2)

    fres_d_sin, fres_d_cos = scipy.special.fresnel(x_delta)
    fres_sum_sin, fres_sum_cos = scipy.special.fresnel(x_sum)

    ret = 1 / (2 * x_delta) * (fres_d_cos * np.cos(theta_delta) +
                               fres_d_sin * np.sin(theta_delta)
                              ) - 1 / (2 * x_sum) * (fres_sum_cos *
                                                     np.cos(theta_sum) +
                                                     fres_sum_sin *
                                                     np.sin(theta_sum))

    if isinstance(ret, np.ndarray):
        return np.real(np.array([0 if np.isnan(x) else x for x in ret]))
    return np.real(0 if np.isnan(ret) else ret)

def redor_sum(x_axis, powder_angles, gamma_angles, myspinsystems,
              dipole_scaling_factor=1, i_atomset_2=1/2, f1=None):
    """
    Calculate the normalized REDOR difference signal. Can handle the following
    cases:

    - Multiple spin system IS_x
    - Motional averaging in the fast motional limit with same probability
      for each step in the trajectory
    - Quadrupolar S nuclei including imperfect excitation of the latters
      outer energy levels

    Parameters
    ----------
    x_axis : np.ndarray
        Mixing times, i.e. your x-axis, in seconds.
    powder_angles : int or array of [[alpha, beta, gamma], ...]
        If int: Number of zcw powder angles to be used for the calculation.
        Gives the number of alpha and beta angles. In this case,
        *gamma_angles* has to be set separately. Beware: not the exact number
        will be used, but the next higher Fibonacci number.
        If array: The explicit angle sets to use.
    gamma_angles : int
        The number of gamma angles to use, which usually can be set
        significantly smaller than the number of alpha and beta angles.
        Ignored if a set of angles is supplied
        to *powder_angles*.
    myspinsystems : list of :class:`morty.atomistic.Spinsystem`
        Spinsystem instance(s) to extract the internuclear vector orientation,
        and the dipolar coupling constant from. If the list holds more than
        one element, the dipole tensors are averaged.
    dipole_scaling_factor : float
        Scale all occuring dipole couplings by the supplied scaling factor. Does D*f.
    i_atomset_2 : float
        Sets the total angular momentum quantum number of the S nucleus. Defaults to 1/2.
    f1 : list of floats
        Correction factor for the dipole couplings for all states, accounts for imperfect π pulses.
        Say I_S = 3/2, you supply [0.9] to scale m_l = -3/2, 3/2 by 0.9. 1/2 will
        not be scaled.

    """
    f1 = np.array(f1)
    mls = np.array([x for x in np.arange(-i_atomset_2, i_atomset_2+1, 1)])
    mls = mls[int(np.ceil(len(mls)/2)):len(mls)] #* f1
    #print('mls ', mls, 'f1', f1)
    dipole_tensors = [[myspinsystems[j].dd_couplings[i][2]
                       for j in range(len(myspinsystems))]
                      for i in range(len(myspinsystems[0].dd_couplings))
                      if myspinsystems[0].dd_couplings[i][0] == 0]
    dim = len(mls)**len(dipole_tensors)
    dcc_scalings = np.array(
        [[mls[int(i*len(mls)/dim*len(mls)**d)]
          for i in range(int(dim/len(mls)**d))] *
         int(dim/len(mls)**(len(dipole_tensors) - d))
         for d in range(len(dipole_tensors))]).T
    intensity_scalings = np.array(
        [[f1[int(i*len(mls)/dim*len(mls)**d)]
          for i in range(int(dim/len(mls)**d))] *
         int(dim/len(mls)**(len(dipole_tensors) - d))
         for d in range(len(dipole_tensors))]).T

    #print('dcc_scalings ', dcc_scalings)
    #print('int scalings ', intensity_scalings)
    if isinstance(powder_angles, int):
        angles_ = zcw(powder_angles)
        ### reorder to get [alpha, beta, gamma]
        angles = np.array([[a[1], a[0], b*2*math.pi]
                           for a in angles_
                           for b in np.linspace(0, 1, gamma_angles)
                          ])
    else:
        angles = powder_angles
    return (
        np.mean([
            1 - np.prod([
                intensity_scalings[m][n]
                for n in range(len(intensity_scalings[m]))]) +
            1/(8*np.pi**2) * 2*np.pi * np.pi * 2*np.pi / len(angles) *
            np.prod([
                intensity_scalings[m][n]
                for n in range(len(intensity_scalings[m]))]) *
            np.sum(
                np.prod([
                    np.cos(4 * math.sqrt(2) *
                           np.mean([
                               dipole_scaling_factor *
                               dipole_tensors[i][j].coupling_constant / 2 / np.pi *
                               dcc_scalings[m][i] *
                               (np.sin(angles[:, 1]) * np.cos(angles[:, 0]) *
                                dipole_tensors[i][j].eigenbasis.T[0, 2] +
                                np.sin(angles[:, 1]) * np.sin(angles[:, 0]) *
                                dipole_tensors[i][j].eigenbasis.T[1, 2] +
                                np.cos(angles[:, 1]) *
                                dipole_tensors[i][j].eigenbasis.T[2, 2]) *
                               ((- np.cos(angles[:, 2]) * np.sin(angles[:, 0]) -
                                 np.cos(angles[:, 1]) * np.cos(angles[:, 0]) *
                                 np.sin(angles[:, 2])) *
                                dipole_tensors[i][j].eigenbasis.T[0, 2] +
                                (np.cos(angles[:, 2]) * np.cos(angles[:, 0]) -
                                 np.cos(angles[:, 1]) * np.sin(angles[:, 0]) *
                                 np.sin(angles[:, 2])) *
                                dipole_tensors[i][j].eigenbasis.T[1, 2] +
                                (np.sin(angles[:, 2]) * np.sin(angles[:, 1])) *
                                dipole_tensors[i][j].eigenbasis.T[2, 2])
                               for j in range(len(dipole_tensors[0]))
                               ], axis=0)[np.newaxis].T * x_axis)
                    for i in range(len(dipole_tensors))
                ], axis=0) * np.sin(angles[:, 1][np.newaxis].T)
                , axis=0)
            for m in range(len(dcc_scalings))], axis=0)
    )

def sq_spinecho(time, coupling_constant, kappa):
    """
    Calculation of the evolution of intensity for SQ recoupling experiments
    with spinechos. :cite:`Mueller1995a`

    Parameters
    ----------
    time : float or array_like
        Time (in s) of the recoupling.
    coupling_constant : float
        Dipolar coupling constant of the nuclei in rad/s.
    kappa : float
        Scaling factor of the pulse sequence (plus nutation and rotation
        frequency).

    Returns
    -------
     intensity : int or np.ndarray
        Relative intensity. This will be a numpy array or a float, depending on
        the type supplied for `t`.

    """
    ret = (np.pi / (2 * np.sqrt(2)) *
           scipy.special.jn(-0.25, 3 / 2 *
                            coupling_constant * time * kappa * 1 + 0j) *
           scipy.special.jn(0.25, 3 / 2 *
                            coupling_constant * time * kappa * 1 + 0j))

    if isinstance(ret, np.ndarray):
        return np.real(np.array([1 if np.isnan(x) else x for x in ret]))
    return np.real(1 if np.isnan(ret) else ret)
