"""
Reads and analyzes processed Bruker NMR measurements. It does not read the FID,
so Fourier transform has to be done in TOPSPIN. All spectrum types inherit the
basic *Spectrum* class and extend its features to the specific needs.

"""

import math
import os
import numpy as np
import lmfit

from . import pseudovoigt
from . import pseudovoigt_integral
from ..util import find_nearest_index_by_value


__all__ = ['Spectrum1D', 'Spectrum2D', 'SpectrumPseudo2D', 'SpectrumAxis', 'Ppm']

#TODO realise a list of PPMs or tuples so multiple PPMs is easier
class Ppm:
    """
    Data type that can be used to specify a range/value in ppm.

    When accessing data from any Spectrum object, you can supply a
    `Ppm` instance instead of :class:`slice`.

    Attributes
    ----------
    high_ppm : float
        The upper bound of the range.
    low_ppm : float
        The lower bound of the range.

    Examples
    --------
    Use with Spectrum1D to define a range you want to display. E.g. ::

        plot(spectrum.axis_f2[Ppm(100, 10)], spectrum[Ppm(100, 10)])

    """
    def __init__(self, high_ppm, low_ppm):
        """
        Instantiates the `Ppm` object.

        Parameters
        ----------
        high_ppm : float
            The higher ppm value.
        low_ppm : float
            The lower ppm value.

        """
        self.high_ppm = high_ppm
        self.low_ppm = low_ppm


class SpectrumAxis(np.ndarray):
    """
    Data type defined to represent a spectrum axis.

    Allows the usage of :class:`morty.analytical.Ppm` with the spectrum axis.

    """
    def __new__(cls, args):
        return np.asarray(args).view(cls)

    def __getitem__(self, boundary):
        if isinstance(boundary, Ppm):
            if boundary.high_ppm is None:
                high_ppm_index = 0
            else:
                high_ppm_index = find_nearest_index_by_value(
                    self.__array__(), boundary.high_ppm, 'lower')
            if boundary.low_ppm is None:
                low_ppm_index = len(self.__array__())
            else:
                low_ppm_index = find_nearest_index_by_value(
                    self.__array__(), boundary.low_ppm, 'higher')
            index = slice(high_ppm_index, low_ppm_index)
        else:
            index = boundary
        return super().__array__()[index]


class Spectrum:
    """
    Meta class for NMR spectra. Do not use.

    All other spectrum classes inherit this.

    """
    def __init__(self, folder, procno, generic_f1):
        """
        Loads a TOPSPIN folder.

        Parameters
        ----------
        folder : str
           The path to the TOPSPIN folder to load.
        procno : int
            The processing number of the spectrum to load.
        generic_f1 : bool
            Use this for pseudo 2D spectra.

        """

        self.acqu_pars = {}
        with open(os.path.join(folder, 'acqus')) as acqu:
            for line in acqu:
                if line.startswith('##$'):
                    # Read in arrays. This is ugly - somebody who knows regexp
                    # should do it properly.
                    if '(0..' in line.split()[1]:
                        # remove two characters from the end because of \n
                        my_array = [None] * (int(line.split('..')[1][:-2]) + 1)
                        varname = line.split()[0][3:-1]
                        read = 0
                        while read < len(my_array) - 1:
                            line = acqu.readline().strip()
                            for element in line.split():
                                my_array[read] = element
                                read += 1
                        self.acqu_pars[varname] = my_array
                    else:
                        self.acqu_pars[line.split()[0][3:-1]] = line.split()[1]

        path = os.path.join(folder, 'pdata', str(procno))

        self.name = None
        self.spc_c = None
        self.base = None

        # load F2 parameters
        self.dim = 1
        self.proc_pars_f2 = {}
        with open(os.path.join(path, 'procs')) as par_file:
            for line in par_file:
                if line.startswith('##$'):
                    self.proc_pars_f2[line.split()[0][3:-1]] = line.split()[1]

        # little/big endian
        if self.proc_pars_f2['BYTORDP'] == '0':
            readdtype = '<i4'
        else:
            readdtype = '>i4'

        # F2 axis
        # Note the endpoint=False.
        self.axis_f2 = SpectrumAxis(np.linspace(
            float(self.proc_pars_f2['OFFSET']),
            float(self.proc_pars_f2['OFFSET']) -
            float(self.proc_pars_f2['SW_p']) /
            float(self.proc_pars_f2['SF']),
            np.uint(self.proc_pars_f2['SI']), endpoint=False))

        # load (optional) F1 parameters and the 1D/2D spectrum
        try:
            with open(os.path.join(path, 'proc2s')) as par_file:
                self.dim = 2
                self.proc_pars_f1 = {}
                for line in par_file:
                    if line.startswith('##$'):
                        self.proc_pars_f1[line.split()[0][3:-1]] = (
                            line.split()[1])

            # spectra are subdivided into sub-matrices
            self.spc = np.zeros((
                int(self.proc_pars_f2['SI']),
                int(self.proc_pars_f1['SI'])))
            block_size = (int(self.proc_pars_f1['XDIM']) *
                          int(self.proc_pars_f2['XDIM']))
            with open(os.path.join(path, '2rr'), 'rb') as data_file:
                for i in range(0, int(int(self.proc_pars_f1['SI']) /
                                      int(self.proc_pars_f1['XDIM']))):
                    for j in range(0, int(int(self.proc_pars_f2['SI']) /
                                          int(self.proc_pars_f2['XDIM']))):
                        # load a XDIM(F2) x XDIM(F1) block from our open file
                        # buffer
                        self.spc[j * int(self.proc_pars_f2['XDIM']):
                                 ((j + 1) * int(self.proc_pars_f2['XDIM'])),
                                 i * int(self.proc_pars_f1['XDIM']):(i + 1) *
                                 int(self.proc_pars_f1['XDIM'])] = (
                                     np.frombuffer(
                                         data_file.read(4 * block_size),
                                         count=block_size,
                                         dtype=readdtype
                                         ).astype(np.int64) * 2 **
                                     int(self.proc_pars_f1['NC_proc']
                                        )).reshape(
                                            (int(self.proc_pars_f1['XDIM']),
                                             int(self.proc_pars_f2['XDIM']))).T

            if (self.proc_pars_f1['OFFSET'] != '0') and (generic_f1 is False):
                self.axis_f1 = SpectrumAxis(np.linspace(
                    float(self.proc_pars_f1['OFFSET']),
                    float(self.proc_pars_f1['OFFSET']) -
                    float(self.proc_pars_f1['SW_p']) /
                    float(self.proc_pars_f1['SF']),
                    np.uint(self.proc_pars_f1['SI']), endpoint=False))
            else:
                self.axis_f1 = SpectrumAxis(
                    np.linspace(0, np.uint32(self.proc_pars_f1['SI']),
                                np.uint32(self.proc_pars_f1['SI'])))
        # IOError: not a 2D spectrum! Load 1D.
        except IOError:
            self.spc = np.fromfile(os.path.join(path, '1r'),
                                   dtype=readdtype).astype(np.int64) * 2 ** \
                int(self.proc_pars_f2['NC_proc'])

    @staticmethod
    def deconvolute_1d(spc, functions, minimizer=None):
        """
        Deconvolutes a spectrum with arbitrary functions.

        Parameters
        ----------
        spc : array
            The intensity values of the spectrum. Since this is a static
            method,  you can use this on any data.
        functions : array of dict
            Each dictionary has the following keys:

                - function : callable
                - params : array of tuples
                    Tuples with starting values and boundaries as required by
                    lmfit have to be supplied: ::

                        (name, start, vary, min, max, expr)

                    where *name* is the name of the parameter in the function
                    declaration, *start* is the starting value, *vary* will not
                    optimize the parameter if False, *min* and *max* are the
                    boundaries and *expr* can be used to apply non-linear
                    constraints (see lmfit documentation). Be aware the
                    parameter names are prefixed with a character, to be able
                    to access parameters of other functions used. That is, the
                    first function is prefixed by 'a', the second by 'b' and so
                    on. If you want the parameter fwhm of first function to be
                    twice that of the second function, you'd have to set expr
                    to '2*bfwhm'.
                - kwargs : dict
                    Static arguments that are given to the function. Please
                    note that you have to supply the x axis to your function
                    yourself! The name of the argument for the axis won't be
                    guessed, so usually you would supply: ::

                        'kwargs' : {'xaxis' : my_axis}

        minimizer : {'nelder', 'lbfgsb', 'powell', 'cg', 'newton', ...}
            Use another minimizer algorithm. See the `lmfit homepage \
            <http://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table>`_
            for available options. A Levenberg-Marquardt will always be
            executed afterwards, since it is the only algorithm that yields
            uncertainties. If not set, a single Levenberg-Marquardt	run will be
            performed.

        Returns
        -------
        values : tuple of dicts
            The fitted values for each function as a dict.
        uncert : tuple of dicts
            Uncertainties of the fitted parameters.
        opt : :class:`lmfit.Minimizer`
            The minimizer object, that holds all information about the fitting.

        Examples
        --------
        Fit a spectrum with a CSA and a pseudo-voigt function: ::

            angles = morty.analytical.zcw(1000)
            deconvolute_1d(my_spc[Ppm(300, -100)],
                           [{'function' : morty.analytical.csa,
                             'params' : (('aniso', -100, True, -50, -150,
                                          None),
                                         ('asym', 0, True, 0, 1, None),
                                         ('iso', 10, True, 0, 30, None),
                                         ('scaling', 1, True, .1, 1.1, None)),
                             'kwargs' : {'powder_angles' : angles,
                                         'omegas' : my_spc.axis_f2[Ppm(300,
                                                                       -100)]}
                            },
                            {'function' :
                             morty.analytical.pseudovoigt,
                             'params' : (('iso', 5, True, 0, 10, None),
                                         ('sigma', 1, True, .1, 3, None),
                                         ('gamma', 1, True, .1, 3, None),
                                         ('eta', .5, True, 0, 1, None),
                                         ('intensity', 1, True, .2, 1.1)),
                             'kwargs' : {'x_axis' : my_spc.axis_f2[Ppm(300,
                                                                   -100)]}
                            }])

        The first part speeds up the calculation by supplying zcw powder
        angles for the CSA calculation.

        If you wanted to add a second Pseudo-Voigt signal with identical FWHM
        and Gauss/Lorentz ratio, you'd use ::

            ('sigma', 1, True, .1, 3, '2sigma'),
            ('gamma', 1, True, .1, 3, '2gamma'),
            ('eta', .5, True, 0, 1, '2eta')

        in its `params` value.

        The optimized values are returned in a list. If you had used only
        Pseudo-Voigt functions, you could easily plot it with ::

            plot(my_spc.axis_f2[Ppm(300, -100)],
                 np.sum(pseudovoigt(
                    **dict(x, **{'x_axis': my_spc.axis_f2[Ppm(300, -100)]}))
                    for x in opt[0])))

        assuming that you wrote the result to `opt`.

        Important properties can be accessed in `opt[2].chisqr` (Chi square),
        `opt[2].success` (if fit converged) and `opt[2].residual`.
        See the `lmfit homepage \
        <http://lmfit.github.io/lmfit-py/fitting.html#fit-results-label>`__
        for details.

        Notes
        -----
        Be careful when using only one function or parameter: (10) is equal to
        10 and can not be iterated. Use (10,).

        Also, it is up to you if you work with a normalized spectrum or not. In
        general it is more convenient to divide the spectrum by its largest
        point max(my_spc), so you can use values between 0 and 1 for the
        intensity of the signals.

        """
        # write Parameters() object with renamed parameter names
        pars = lmfit.Parameters()
        for i in range(len(functions)):
            # We encode the number of function as a character, starting with a.
            # This is due to the fact, that parameters/variables cannot start
            # with a number.
            pars.add_many(*tuple((str(i) + functions[i]['params'][j][0],
                                  functions[i]['params'][j][1],
                                  functions[i]['params'][j][2],
                                  functions[i]['params'][j][3],
                                  functions[i]['params'][j][4],
                                  functions[i]['params'][j][5])
                                 for j in range(len(functions[i]['params']))))

        # this functions calls each supplied function with the given parameters
        # and calculates the total deviation
        def complete_deviation(arguments, spc, functions):
            """
            Function that returns the deviation, used by the optimizer.

            """
            my_args = arguments.valuesdict()
            deviation = np.copy(spc)
            for i, myfunction in enumerate(functions):
                deviation -= myfunction['function'](
                    **dict({myfunction['params'][j][0]:
                            my_args[chr(97 + i) + myfunction['params'][j][0]]
                            for j in range(len(myfunction['params']))},
                           **functions[i]['kwargs']))
            return deviation

        # Use another minimizer first, if requested.
        if minimizer is not None:
            premin = lmfit.minimize(complete_deviation, pars,
                                    args=(spc, functions), method=minimizer)
            pars = premin.params
        opt = lmfit.minimize(complete_deviation, pars, args=(spc, functions))

        results, uncert = [None] * len(functions), [None] * len(functions)
        for i in range(len(functions)):
            results[i] = {functions[i]['params'][j][0]:
                          opt.params[chr(97 + i) +
                                     functions[i]['params'][j][0]].value
                          for j in range(len(functions[i]['params']))}
            uncert[i] = {functions[i]['params'][j][0]:
                         opt.params[chr(97 + i) +
                                    functions[i]['params'][j][0]].stderr
                         for j in range(len(functions[i]['params']))}
        return results, uncert, opt


class Spectrum1D(Spectrum):
    """
    Reading and analysis of a 1D spectrum.

    Attributes
    ----------
    spc : np.ndarray
        The "raw" spectrum, meaning as-read from the TOPSPIN folder.
    spc_c : np.ndarray
        The baseline-corrected spectrum, meaning the baseline-correction
        has been performed within this object.
    axis_f2 : :class:`morty.analytical.SpectrumAxis`
        The F2-axis in ppm.
    acqu_pars : dict
        Holds information about the acquisition, like frequency offsets.
    proc_pars_f2 : dict
        Holds information about the processed spectrum, e.g. the frequency
        scale.
    dim : int
        The dimension of the spectrum. Since this class handles 1D spectra, this should be 1
    base : np.ndarray
        The calculated baseline of the spectrum. This is only set after a baseline correction has been performed.

    Notes
    -----
    - The Spectrum1D instance will work with baseline corrected data in the
      `spc_c` attribute if correction has been performed, otherwise `spc`
      will be used.
    - If you iterate over the object, you will iterate over all points in the
      spectrum
    - Slicing:
        - ``spectrum[12:15]`` yields a slice from index 12 to (including)
          14 as usual
        - ``spectrum[Ppm(14:18)]`` yields the spectrum from 14 ppm to 18 ppm

    Examples
    --------
    You can easily read in a TOPSPIN folder and plot the 1D: ::

        myspc = Spectrum1D('myfolder')
        plot(myspc.axisf2, mySpc)

    """

    def __init__(self, folder=None, procno=1, spc=None, proc_pars_f2=None,
                 spc_c=None, acqu_pars=None, axis_f2=None):
        """
        Instantiates a 1D Spectrum.

        The 1D spectrum can be constructed either by providing the TOPSPIN
        measurement folder from which to read or can be constructed using
        explicit data. Usually a user would supply the path of the folder.

        Parameters
        ----------
        folder : str, optional
            Path of the Bruker experiment folder.
        procno : int, optional
            Number of measurement inside the given folder.
        spc : np.ndarray, optional
            The "raw" spectrum to use.
        proc_pars_f2 : dict, optional
            TOPSPIN processing parameters for F2 to construct `spc` from.
        spc_c : np.array, optional
            The baseline corrected spectrum to use. Usually this is only
            constructed when the baseline correction is performed with
            `morty`.
        acqu_pars : dict, optional
            TOPSPIN aquisition parameters.
        axis_f2 : :class:`morty.analytical.SpectrumAxis`, optional
            The F2 axis to use.

        """
        if spc is None and folder is not None:
            super().__init__(folder, procno, False)
        else:
            self.spc = spc
            self.spc_c = spc_c
            self.proc_pars_f2 = proc_pars_f2
            self.dim = 1
            self.acqu_pars = acqu_pars
            if axis_f2 is None:
                self.axis_f2 = SpectrumAxis(np.linspace(
                    float(self.proc_pars_f2['OFFSET']),
                    float(self.proc_pars_f2['OFFSET']) -
                    float(self.proc_pars_f2['SW_p']) /
                    float(self.proc_pars_f2['SF']),
                    np.uint(self.proc_pars_f2['SI']), endpoint=False))
            else:
                self.axis_f2 = axis_f2
            self.base = None

    def __array__(self):
        return self.spc if self.spc_c is None else self.spc_c

    def __getitem__(self, boundary):
        if isinstance(boundary, Ppm):
            if boundary.high_ppm is None:
                high_ppm_index = 0
            else:
                high_ppm_index = find_nearest_index_by_value(
                    self.axis_f2, boundary.high_ppm, 'lower')
            if boundary.low_ppm is None:
                low_ppm_index = len(self.axis_f2)
            else:
                low_ppm_index = find_nearest_index_by_value(
                    self.axis_f2, boundary.low_ppm, 'higher')
            index = slice(high_ppm_index, low_ppm_index)
        else:
            index = boundary
        return self.spc[index] if self.spc_c is None else self.spc_c[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self.spc[i] if self.spc_c is None else self.spc_c[i]

    def __len__(self):
        return len(self.spc)

    def baseline(self, f2range=((0, None),), deg=2):
        """
        Baseline correction for the spectrum.

        Performs a polynomial fit to the spectrum and saves a corrected
        spectrum to `spc_c` and the polynom function used as
        a basis in `base`.

        Parameters
        ----------
        f2range : tuple of tuples or :class:`morty.analytical.Ppm`
            List of upper/lower bounds for the areas used to fit the baseline.
            None as value for the second value of the tuple equals to the
            maximum value.
        deg : int
            Degree of the polynomial used to fit the baseline.

        Examples
        --------
        To perform a baseline fit, you select one or more areas in your
        spectrum only containing noise and perform the fit on these areas. We
        will use a range from 120 to 80 and from 0 to -30 ppm. ::

            myspec = Spectrum1D(myfolder)
            myspec.baseline(f2range=[Ppm(120, 80), Ppm(0, -30)])

        """

        self.base = np.zeros(len(self.spc))

        xfit = []
        for myrange in f2range:
            if isinstance(myrange, tuple):
                xfit.extend(list(range(myrange[0], myrange[1] if myrange[1] is not None
                                       else len(self.spc))))
            if isinstance(myrange, Ppm):
                xfit.extend(list(range(np.where(self[myrange][0] == self[:])[0][0],
                                       np.where(self[myrange][-1] == self[:])[0][0]))
                           )

        myfit = np.poly1d(np.polyfit(xfit, self.spc[xfit], deg))
        self.base = myfit(np.linspace(0, len(self.spc) - 1, len(self.spc)))

        self.spc_c = self.spc - self.base

    def baseline_subtract_measurement(self, spectrum_bg, scaling=1):
        """
        Subtracts a background measurement from the current spectrum.

        This will overwrite any existent baseline correction applied before,
        meaning the background measurement's *spc* is subtracted from the
        spectrums *spc*.

        Parameters
        ----------
        spectrum_bg : :class:`morty.analytical.Spectrum1D`
            The background measurement.
        scaling : float
            scaling factor to apply to the background.

        Examples
        --------
        To subtract the background measurement, you read in your original
        measurement as usual, and then the background measurement. ::

            myspec = Spectrum1D(myfolder)
            mybg = Spectrum1D(myfolder)
            myspec.baseline_subtract_measurement(mybg)

        """
        self.spc_c = self.spc - spectrum_bg.spc * scaling

    def integrate_by_sum(self, int_range=None):
        """
        Integrate spectrum in a certain range by summing it up.

        Integrates spectrum in a certain range and returns a the absolute
        values.

        Parameters
        ----------
        int_range : :class:`slice(min:max)` or :class:`morty.analytical.Ppm`, optional
            F2 range for the integration. If None, the whole spectrum is
            integrated.

        Returns
        -------
        integral : float
            The sum over the specified range.

        """
        if int_range is None:
            int_range = slice(0, len(self))
        elif not isinstance(int_range, slice) and not isinstance(int_range,
                                                                 Ppm):
            int_range = slice(int_range[0], int_range[1])
        return np.sum(self[int_range], axis=0)

    @staticmethod
    def integrate_deconvoluted(spc, axis, signals, minimizer=None):
        """
        Deconvolutes and integrates a pseudo 1D spectrum using a sum of Pseudo
        Voigt profiles.

        Parameters
        ----------
        spc : array_like
            Spectrum data.
        axis : array_like
            Spectrum axis.
        signals : tuple of tuples of tuples
            For each signal to be fitted, a list of the parameters *iso*,
            *sigma*, *gamma*, *intensity* and *eta* has to be supplied in the
            form ::

                (('parameter', start_value, vary, min, max, expr), ...)

            where *vary* must be True, if the parameter is to be optimized and
            *expr* can be used to use shared parameters (see example). When using
            *expr*, the parameter names are prefixed by a character indicating
            which signal it belongs to, e.g. parameters of the first signal are
            prefixed by 'a', the second by 'b' and so on.
        minimizer : 'nelder', 'lbfgsb', 'powell', 'cg', 'newton', ..., optional
            Use another minimizer algorithm. See
            http://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table for
            available options. A Levenberg-Marquardt will always be executed
            afterwards, since it is the only algorithm that yields
            uncertainties. If not set, a single Levenberg-Marquardt	run will be
            performed.

        Returns
        -------
        integrals : tuple
            The integral of each signal, scaled to the original scale of the
            experimental spectrum.
        uncert : list
            Uncertainty for each integral.
        minimizer : (values, uncertainties, minimizer)
            The return value of :class:`morty.analytical.Spectrum.deconvolute_1d()`.

        Examples
        --------
        The following code will deconvolute a 1D spectrum with two signals,
        which share the same FWHM for Gauss and Lorentz part and the
        Gauss/Lorentz ratio: ::

            signals = ((('iso', 5, True, 4, 6, None),
                        ('sigma', 1, True, .5, 2, None),
                        ('gamma', 1, True, .5, 2, None),
                        ('eta', .5, True, 0, 1, None),
                        ('intensity', 1, True, 0.1, 1.1, None)),
                       (('iso', 10, True, 9, 11, None),
                        ('sigma', 1, True, .5, 2, '1sigma'),
                        ('gamma', 1, True, .5, 2, '1gamma'),
                        ('eta', .5, True, 0, 1, '1eta'),
                        ('intensity', 1, True, 0.1, 1.1, None))
            Spectrum1D.integrate_deconvoluted(my_spc.spc / max(my_spc.spc),
                                              my_spc.axis_f2, signals)

        """
        functions = [{'function': pseudovoigt,
                      'params': signals[i], 'kwargs': {'x_axis': axis}}
                     for i in range(len(signals))]
        opt = Spectrum.deconvolute_1d(spc, functions, minimizer)
        integ = tuple(pseudovoigt_integral(opt[0][i]['sigma'],
                                           opt[0][i]['gamma'],
                                           opt[0][i]['intensity'],
                                           opt[0][i]['eta'])
                      for i in range(len(opt[0])))
        # calculate uncertainties
        uncert = [None] * len(opt[0])
        for i in range(len(opt[0])):
            # sqrt(pi/ln(2)) = 2.1289340388624525
            if (opt[1][i]['sigma'] is not None and
                opt[1][i]['eta'] is not None and
                opt[1][i]['intensity'] is not None and
                opt[1][i]['gamma'] is not None):
                d_sigma = (opt[0][i]['intensity'] * opt[0][i]['eta'] *
                           2.1289340388624525 * opt[1][i]['sigma']) ** 2
                d_gamma = ((1 - opt[0][i]['eta']) * opt[0][i]['intensity'] *
                           np.pi / 2 * opt[1][i]['gamma']) ** 2
                d_int = ((opt[0][i]['eta'] * opt[0][i]['sigma'] / 2 *
                          2.1289340388624525 +
                          (1 - opt[0][i]['eta']) * opt[0][i]['gamma'] *
                          np.pi / 2) *
                         opt[1][i]['intensity']) ** 2
                d_eta = ((opt[0][i]['intensity'] * opt[0][i]['gamma'] / 2 *
                          2.1289340388624525 - opt[0][i]['intensity'] *
                          opt[0][i]['gamma'] * np.pi / 2) * opt[1][i]['eta']) ** 2
                uncert[i] = math.sqrt(d_sigma + d_gamma + d_int + d_eta)
            else:
                uncert[i] = None
        return [integ, uncert, opt]


class Spectrum2D(Spectrum):
    """
    Reading and analysis of a 2D spectrum.

    Attributes
    ----------
    spc : np.ndarray
        The "raw" spectrum, meaning as-read from the TOPSPIN folder.
    spc_c : np.ndarray
        The baseline-"corrected" spectrum, if baseline correction has been
        performed.
    axis_f2 : :class:`morty.analytical.SpectrumAxis`
        The F2 axis in ppm.
    axis_f1 : :class:`morty.analytical.SpectrumAxis`
        The F1 axis.
    acqu_pars : dict
        Holds information about the acquisition, like frequency offsets.
    proc_pars_f1 : dict
        Holds information about the F1 dimension of the processed spectrum,
        e.g. the frequency scale.
    proc_pars_f2 : dict
        Same as `proc_pars_f1`, but for the F2 dimension.
    dim : int
        The dimension of the spectrum. Since this class handles 2D spectra, this should be 2.
    base : np.ndarray
        The calculated baseline of the spectrum. This is only set after a baseline correction has been performed.

    Notes
    -----
    The Spectrum2D instance will work with baseline corrected data in
    `spc_c` if correction has been performed, otherwise `Spectrum2D.spc`
    will be used.

    Examples
    --------
    You can slice this sort of spectrum just like any other. Be aware that
    you have to hand over the ppm values for F2 first for spc, just like you
    would have to when using indices directly: ::

        myspc2D = Spectrum2D(myfolder)
        ppmf2 = [15, -5]
        ppmf1 = [20, -10]
        nue1, nue2 = np.meshgrid(myspc2D.axis_f1[Ppm(ppmf1[0], ppmf1[1])],
                                 myspc2D.axis_f2[Ppm(ppmf2[0], ppmf2[1])])
        plot_spc = myspc2D[Ppm(ppmf2[0], ppmf2[1]),
                           Ppm(ppmf1[0], ppmf1[1])]

    """

    def __init__(self, folder=None, procno=1, spc=None, proc_pars_f2=None,
                 proc_pars_f1=None, spc_c=None, acqu_pars=None):
        """
        Set up an instance of a 2D Spectrum.

        The 2D spectrum can be constructed either by providing the TOPSPIN
        measurement folder from which to read or can be constructed using
        explicit data. Usually a user would supply the path of the folder.

        Parameters
        ----------
        folder : str, optional
            Path of the bruker experiment folder.
        procno : int, optional
            Number of measurement inside the given folder.
        spc : array, optional
            If supplied, the object will be initialized with data from `spc`,
            `spc_c`, `proc_pars_f2`, `proc_pars_f1` and `acqu_pars`. This is
            useful when creating copies of existing Spectrum2D instances, but
            is not needed usually.
        proc_pars_f2 : dict, optional
        proc_pars_f1 : dict, optional
        spc_c : array, optional
        acqu_pars : dict, optional

        """
        if spc is None and folder is not None:
            super().__init__(folder, procno, False)
        else:
            self.spc = spc
            self.spc_c = spc_c
            self.proc_pars_f2 = proc_pars_f2
            self.dim = 2
            self.acqu_pars = acqu_pars
            self.axis_f2 = SpectrumAxis(np.linspace(
                float(self.proc_pars_f2['OFFSET']),
                float(self.proc_pars_f2['OFFSET']) -
                float(self.proc_pars_f2['SW_p']) /
                float(self.proc_pars_f2['SF']),
                np.uint(self.proc_pars_f2['SI']), endpoint=False))
            self.proc_pars_f1 = proc_pars_f1
            self.axis_f1 = SpectrumAxis(np.linspace(
                float(self.proc_pars_f1['OFFSET']),
                float(self.proc_pars_f1['OFFSET']) -
                float(self.proc_pars_f1['SW_p']) /
                float(self.proc_pars_f1['SF']),
                np.uint(self.proc_pars_f1['SI']), endpoint=False))

    def __array__(self):
        return self.spc if self.spc_c is None else self.spc_c

    def __getitem__(self, boundary):
        if isinstance(boundary[0], Ppm):
            if boundary[0].high_ppm is None:
                high_ppm_index_f2 = 0
            else:
                high_ppm_index_f2 = find_nearest_index_by_value(
                    self.axis_f2, boundary[0].high_ppm, 'lower')
            if boundary[0].low_ppm is None:
                low_ppm_index_f2 = len(self.axis_f2)
            else:
                low_ppm_index_f2 = find_nearest_index_by_value(
                    self.axis_f2, boundary[0].low_ppm, 'higher')
            index_f2 = slice(high_ppm_index_f2, low_ppm_index_f2)
        else:
            index_f2 = boundary[0]

        if isinstance(boundary[1], Ppm):
            if boundary[1].high_ppm is None:
                high_ppm_index_f1 = 0
            else:
                high_ppm_index_f1 = find_nearest_index_by_value(
                    self.axis_f1, boundary[1].high_ppm, 'lower')
            if boundary[1].low_ppm is None:
                low_ppm_index_f1 = len(self.axis_f1)
            else:
                low_ppm_index_f1 = find_nearest_index_by_value(
                    self.axis_f1, boundary[1].low_ppm, 'higher')
            index_f1 = slice(high_ppm_index_f1, low_ppm_index_f1)
        else:
            index_f1 = boundary[1]

        return (self.spc[index_f2, index_f1] if self.spc_c is None
                else self.spc_c[index_f2, index_f1])


    def integrate(self, limit1, limit2):
        """
        Integrate an area of the spectrum.

        Integrates in the simplest way: adding up all intensities within a
        certain range.

        Parameters
        ----------
        limit1 : tuple(min, max)
            Upper and lower limit in F1.
        limit2 : tuple(min, max)
            Upper and lower limit in F2.

        Returns
        -------
        integral : float
            Integrated intensity.

        """

        summed_spec = 0
        for row in range(limit2[0], limit2[1]):
            if self.spc_c is not None:
                summed_spec += np.sum(self.spc_c[row, limit1[0]:limit1[1]])
            else:
                summed_spec += np.sum(self.spc[row, limit1[0]:limit1[1]])
        return summed_spec

    def baseline(self, f2range=((0, None),), f1range=None, deg=2):
        """
        Baseline correction for the spectrum.

        Performs a polynomial fit to the spectrum and saves a corrected
        spectrum to `Spectrum2D.spc_c` and the polynom function used as a
        basis in `Spectrum2D.base`.

        Parameters
        ----------
        f2range : tuple
            List of tuples with upper/lower bounds in the F2 dimension for the
            areas used to fit the baseline. None as value for the second value
            of the tuple equals to the maximum value. Values are given in ppm
        f1range : tuple
            List of tuples upper/lower bounds in the F1 dimension for the areas
            used to fit the baseline. None as value for the second value of the
            tuple equals to the maximum value. None as value for the parameter
            itself skips the baselinecorrection in the F1 dimension. Values
            are given in ppm.
        deg : int
            Degree of the polynom used to fit the baseline.

        """
        self.base = np.zeros((
            len(self.spc[:, 0]), len(self.spc[0, :])))

        xfit = []
        for myrange in f2range:
            if isinstance(myrange, tuple):
                xfit.extend(list(range(myrange[0], myrange[1] if myrange[1] is not None
                                       else len(self.spc[:, 0]))))
            if isinstance(myrange, Ppm):
                xfit.extend(list(range(np.where(self[myrange, 0][0] == self[:, 0])[0][0],
                                       np.where(self[myrange, 0][-1] == self[:, 0])[0][0]))
                           )

        # fit in f2 dimension
        for i in range(0, len(self.spc[0])):
            myfit = np.poly1d(np.polyfit(xfit, self.spc[xfit, i], deg))
            self.base[:, i] = myfit(np.linspace(0, len(self.spc[:, i]) - 1,
                                                len(self.spc[:, i])))

        self.spc_c = self.spc - self.base

        # do a fit in the F1 dimension. Use the already corrected spectrum
        # spc_c for the fit, add the fit to base and redefine spc_c
        if f1range is not None:
            xfit = []
            for myrange in f1range:
                if isinstance(myrange, tuple):
                    xfit.extend(list(range(myrange[0], myrange[1] if myrange[1] is not None
                                           else len(self.spc[0, :]))))
                if isinstance(myrange, Ppm):
                    xfit.extend(list(range(np.where(self[0, myrange][0] == self[0, :])[0][0],
                                           np.where(self[0, myrange][-1] == self[0, :])[0][0])))

            for i in range(0, len(self.spc[:, 0])):
                myfit = np.poly1d(np.polyfit(xfit, self.spc_c[i, xfit], deg))
                self.base[i, :] = myfit(np.linspace(0, len(self.spc[i, :]) -
                                                    1, len(self.spc[i, :])))

            self.spc_c = self.spc - self.base

    def symmetric_spectrum(self):
        """
        Returns a symmetrized spectrum.

        Use this e.g. for static 2D exchange spectra, which should be
        symmetric with the diagonal. Use this on quadratic spectra, that means
        the SI in topspin should be the same for both dimensions.

        Returns
        -------
        spectrum : :class:`morty.analytical.Spectrum2D`

        """
        return Spectrum2D(acqu_pars=self.acqu_pars,
                          proc_pars_f1=self.proc_pars_f1,
                          proc_pars_f2=self.proc_pars_f2,
                          spc=(self.spc + self.spc.T) / 2,
                          spc_c=None if self.spc_c is None else
                          (self.spc_c + self.spc_c.T) / 2)


class SpectrumPseudo2D(Spectrum):
    """
    Holds a Pseudo 2D Spectrum.

    Attributes
    ----------
    spc : np.ndarray
        The "raw" spectrum, meaning as-read from the TOPSPIN folder.
    spc_c : np.ndarray
        The baseline-"corrected" spectrum, meaning the baseline-correction
        has been performed within this object.
    axis_f2 : :class:`morty.analytical.SpectrumAxis`
        The F2-axis in ppm.
    acqu_pars : dict
        Holds information about the acquisition, like frequency offsets.
    proc_pars_f2 : dict
        Holds information about the processed spectrum, e.g. the frequency
        scale.
    vp_list : list
        If the object is initialized with `load_vp` = True, the vp (pulse length)
        list will be saved to this variable.
    vd_list : list
        If the object is initialized with `load_vd` = True, the vd (delay length)
        list will be saved to this variable.
    vc_list : listt
        If the object is initialized with `load_vc` = True, the vc (counter)
        list will be saved to this variable.
    base : np.ndarray
        The calculated baseline of the spectrum. This is only set after a baseline correction has been performed.

    Notes
    -----
    - The SpectrumPseudo2D instance will work with baseline corrected data in
      `spc_c` if correction has been performed, otherwise `spc` will be used.
    - Iterating over the object, or accessing it with an index or slice will
      return instances (or a list) of `Spectrum1D`.

    """
    def __init__(self, folder, procno=1, load_vd=False, load_vp=False,
                 load_vc=False, num_experiments=None, everynth=(0, 1)):
        """
        Set up an instance of a Pseudo2D Spectrum.

        Parameters
        ----------
        folder : str
            Path of the bruker experiment folder.
        procno : int
            Number of measurement inside the given folder.
        load_vd : bool
            Load a vd (variable delay) list used in the experiment, if it exists.
        load_vp : bool
            Load a vp (variable pulse) list used in the experiment, if it exists.
        load_vc : bool
            Load a vc (variable counter) list used in the experiment, if it exists.
        num_experiments: int
            The number of experiments to read in. Topspin only create datasets
            with a length of multiples of 2, so there will be experiments at
            the end which are actually empty.
        everynth : list of int
            Which subset of spectra to use, e.g. [0, 2] to use every second
            experiment and start counting from index 0, i.e. the first
            spectrum. This comes in handy if e.g. for REDOR reference and
            experiment have been recorded in the same experiment.

        """
        super().__init__(folder, procno, True)
        if num_experiments is not None:
            self.spc = self.spc[:, everynth[0]:num_experiments:everynth[1]]
        else:
            self.spc = self.spc[:, everynth[0]:int(self.proc_pars_f1['TDeff']):
                                everynth[1]]

        if load_vd is True:
            self.vd_list = np.array(open(os.path.join(folder, 'vdlist')
                                        ).read().splitlines(),
                                    dtype=float)[everynth[0]::everynth[1]]
        if load_vp is True:
            self.vp_list = np.array(open(os.path.join(folder, 'vplist')
                                        ).read().splitlines(),
                                    dtype=float)[everynth[0]::everynth[1]]
        if load_vc is True:
            self.vc_list = np.array(open(os.path.join(folder, 'vclist')
                                        ).read().splitlines(),
                                    dtype=float)[everynth[0]::everynth[1]]
        self.base = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [Spectrum1D(spc=self.spc[:, i],
                               spc_c=None if self.spc_c is None
                               else self.spc_c[:, i],
                               proc_pars_f2=self.proc_pars_f2,
                               acqu_pars=self.acqu_pars,
                               axis_f2=self.axis_f2)
                    for i in range(len(self.spc.T[index]))]
        return Spectrum1D(spc=self.spc[:, index],
                          spc_c=None if self.spc_c is None
                          else self.spc_c[:, index],
                          proc_pars_f2=self.proc_pars_f2,
                          acqu_pars=self.acqu_pars,
                          axis_f2=self.axis_f2)

    def __iter__(self):
        for i in range(len(self.spc.T)):
            yield Spectrum1D(spc=self.spc[:, i],
                             spc_c=self.spc_c[:, i]
                             if self.spc_c is not None else None,
                             proc_pars_f2=self.proc_pars_f2,
                             acqu_pars=self.acqu_pars,
                             axis_f2=self.axis_f2)

    def __len__(self):
        return len(self.spc.T)

    def baseline(self, f2range=((0, None),), deg=2):
        """
        Baseline correction for each of the spectra.

        Performs a polynomial fit to the spectra and saves the corrected
        spectra to `spc_c` and the polynomial function values used
        as a basis in `base`.

        Parameters
        ----------
        f2range : tuple of tuples or :class:`morty.analytica.Ppm`
            List with upper/lower bounds in the F2 dimension for the
            areas used to fit the baseline. None as value for the second value
            of the tuple equals to the maximum value.
        deg : int
            Degree of the polynomial used to fit the baseline.

        """
        self.base = np.zeros((
            len(self.spc[:, 0]), len(self.spc[0, :])))

        xfit = []
        for myrange in f2range:
            if isinstance(myrange, tuple):
                xfit.extend(list(range(myrange[0], myrange[1] if myrange[1] is not None
                                       else len(self.spc[:, 0]))))
            if isinstance(myrange, Ppm):
                xfit.extend(list(range(np.where(self[0][myrange][0] ==
                                                self[0][:])[0][0],
                                       np.where(self[0][myrange][-1] ==
                                                self[0][:])[0][0])))
        # fit in f2 dimension
        for i in range(0, len(self.spc[0])):
            myfit = np.poly1d(np.polyfit(xfit, self.spc[xfit, i], deg))
            self.base[:, i] = myfit(np.linspace(0, len(self.spc[:, i]) - 1,
                                                len(self.spc[:, i])))

        self.spc_c = self.spc - self.base

    def baseline_subtract_measurement(self, spectrum_bg, scaling=1):
        """
        Subtracts a background measurement from the current spectrum.

        This will overwrite any existent baseline correction applied before,
        meaning the background measurement's *spc* is subtracted from the
        spectrums *spc*.

        Parameters
        ----------
        spectrum_bg : SpectrumPseudo2D instance
            The background measurement.
        scaling :
            scaling factor to apply to the background.

        Examples
        --------
        To subtract the background measurement, you read in your original
        measurement as usual, and then the background measurement. ::

            myspec = SpectrumPseudo2D(myfolder)
            mybg = SpectrumPseudo2D(myfolder)
            myspec.baseline_subtract_measurement(mybg)

        """
        self.spc_c = self.spc - spectrum_bg.spc * scaling

    def integrate_deconvoluted(self, signals, spc_slice=None, start_spc=0,
                               minimizer=None):
        
        #TODO Does this return intensities or integrals?
        """
        Deconvolutes and integrates a pseudo 2D spectrum.

        This performs a fit of one spectrum (given by `start_spc`) and then
        only fits the intensities for all remaining spectra.

        Parameters
        ----------
        signals : tuple of tuples of tuples
            For each signal to be fitted, a list of the parameters *iso*,
            *sigma*, *gamma*, *intensity* and *eta* has to be supplied in the
            form ::

                [('parameter', start_value, vary, min, max, expr), ...]

            where vary must be True, if the parameter is to be optimized and
            expr can be used to use shared parameters (see example). When using
            expr, the parameter names are prefixed by a character indicating
            which signal it belongs to, e.g. parameters of the first signal are
            prefixed by 'a', the second by 'b' and so on.
        spc_slice : :class:`slice` or :class:`morty.analytical.Ppm`
            If set, fitting will only be performed in this range.
        start_spc : int
            The index of the spectrum to use for the first deconvolution, which
            will serve as starting point for the following deconvolutions in
            the series.
        minimizer : {'nelder', 'lbfgsb', 'powell', 'cg', 'newton', ...}
            Use another minimizer algorithm. See
            http://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table for
            available options. A Levenberg-Marquardt will always be executed
            afterwards, since it is the only algorithm that yields
            uncertainties. If not set, a single Levenberg-Marquardt	run will be
            performed.

        Returns
        -------
        integrals : tuple
            The integrals for each signal and spectrum, scaled to the original
            scale of the experimental spectrum, not 1 as the intensity
            internally in the optimizer.
        uncertanties : list
            Uncertanties for the integrals.
        scaling : float
            Scaling factor used to normalize the spectra. Multiply it with the
            integrals (and uncertanties) to obtain non normalized values.
        minimizer : tuple
            The return values of each call of
            :class:`morty.analytical.Spectrum1D.integrate_deconvoluted()`.

        Notes
        -----
        The spectra will be scaled to the highest intensity in the given range
        of the start_spc. I.e. one peak in start_spc will have an intensity
        around 1. For a buildup curve, where you would use the spectrum with
        the highest intensity as start_spc, your boundaries should look like
        0, 1.5.

        Examples
        --------
        The following code deconvolutes a Pseudo2D spectrum with one signal: ::

            signals = ((('iso', 133, True, 130, 136, None),
                        ('sigma', 1, True, .1, 5, None),
                        ('gamma', 1, True, .1, 5, None),
                        ('intensity', 1, True, 0, 1.5, None),
                        ('eta', .5, True, 0, 1, None)),)
            spc.integrate_deconvoluted(signals, Ppm(20, 0), start_spc=10)

        This will perform the fit between 20 ppm and 0 ppm and will use
        spectrum #10 to determine the FWHM and Gauss/Lorentz ratio.

        """
        if spc_slice is None:
            spc_slice = slice(0, len(self[0]))
        scaling = max(self[start_spc][spc_slice])

        intensities, uncertainties, opts = ([None] * len(self),
                                            [None] * len(self),
                                            [None] * len(self))

        # First run with first spectrum.
        first_spc = Spectrum1D.integrate_deconvoluted(self[start_spc][spc_slice
                                                                     ] /
                                                      scaling,
                                                      self.axis_f2[spc_slice],
                                                      signals,
                                                      minimizer=minimizer)
        intensities[start_spc] = first_spc[0]
        uncertainties[start_spc] = first_spc[1]
        opts[start_spc] = first_spc[2][0]

        # Update functions array. Note that we need to convert the parameters
        # to an array - we can't edit tuples.
        signals = np.array(signals)
        for i, signal in enumerate(signals):
            for j, param in enumerate(signal):
                param = np.array(param)
                if (param[0] == 'iso' or param[0] == 'sigma' or
                        param[0] == 'gamma' or param[0] == 'eta'):
                    # Set fitted value.
                    param[1] = first_spc[2][0][i][param[0]]
                    # vary => False
                    param[2] = False
                signals[i][j] = param

        for spcnr, spc in enumerate(self):
            if spcnr == start_spc:
                continue
            opt = Spectrum1D.integrate_deconvoluted(spc[spc_slice] /
                                                    scaling,
                                                    self.axis_f2[spc_slice],
                                                    signals,
                                                    minimizer=minimizer)
            intensities[spcnr] = opt[0]
            uncertainties[spcnr] = opt[1]
            opts[spcnr] = opt[2][0]
        return intensities, uncertainties, scaling, opts

    def integrate_by_sum(self, int_range=None):
        """
        Integrate each spectrum in a certain range by summing it up.

        Integrates each spectrum in a certain range and returns a list of the
        absolute values.

        Parameters
        ----------
        int_range : slice(min:max) or :class:`morty.analytical.Ppm`
            F2 range for the integration.

        Returns
        -------
        integrals : list of floats
            List of integrals over the specified range for each spectrum.

        """
        return [spc.integrate_by_sum(int_range) for spc in self]
