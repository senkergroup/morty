"""
Sets up simulations (SIMPSON).

Wrappers to call a simulation routine, e.g. SIMPSON or any analytical function.
"""

import math
import warnings
import tempfile
import os.path
import subprocess
import copy
import numpy as np
import lmfit

from ..atomistic import Spinsystem


__all__ = ['SimpsonCaller', 'SimpsonCalculation', 'SpinsystemCreator']


class SimpsonCaller:
    """
    Starts a simpson simulation for the given spinsystem.

    """

    def __init__(self, spinsystem=None, cell_modeller=None, spinsystem_creator=None,
                 body=None, sim_file='simpson.in', calc_dir=None, include_csa=False,
                 include_dipole=True, include_dipole_euler=True,
                 include_quadrupolar=False, result_extension='.fid',
                 simpson_executable='simpson', results=None):
        """
        Initialize the SimpsonCaller class allowing you setup a SIMPSON
        simulation from a :class:`morty.atomistic.Spinsytem`.

        Parameters
        ----------
        body: str
            Filename of an exemplary simpson file. Should contain everything
            apart from the spinsystem section. The spinsystem section should
            look like: ::

                spinsys { [[spinsystem]] }

        sim_file: str
            The filename of the simpson input file written out. If you use the
            raster, we will prepend the step number. We expect this to end in
            '.in'.
        calc_dir: str
            The folder in which the files are stored and the calculations
            performed. If 'temp', a temporary folder will be used.
        include_csa: bool
            If to include the csa in the simulation.
        include_quadrupolar: bool
            If to include the quadrupolar coupling in the simulation. All
            quadrupolar coupling found in the *magres* file are considered.
        include_dipole: bool
            If to include the dipolar coupling in the simulation.
        include_dipole_euler: bool
            If to include the Euler angles for the dipolar coupling
            in the simulation.
        result_extension: str
            We expect the result to be written to a file with the same name
            as the input, but with another file extension (e.g. '.fid').
            Include the dot.
        simpson_executable : string
            The SIMPSON executable to use, e.g. 'simpson'
        results : list
            Used only for a raster. In case you already have (some) results, you
            can supply a list. Steps not calculated yet should contain :const:`None`.

        """
        self.spinsystem = spinsystem
        # For now, this is a list. We might extend this class to allow multiple
        # calculations. If this would be useful is not clear, but maybe we
        # don't want to create the ssh connection again (hugh, we might connect
        # in SimpsonConfig()).
        self.processes = []

        self.job = None

        self.include_csa = include_csa
        self.body_template = body
        self.sim_file = sim_file
        self.calc_dir = calc_dir
        if self.calc_dir is None:
            self.calc_dir = './'
        if self.calc_dir == 'temp':
            self.calc_dir = tempfile.gettempdir()
        self.include_quadrupolar = include_quadrupolar
        self.include_dipole = include_dipole
        self.include_dipole_euler = include_dipole_euler
        self.result_extension = result_extension
        self.simpson_executable = simpson_executable
        self.results = results
        self.cell_modeller = cell_modeller
        self.spinsystem_creator = spinsystem_creator
        self.simpson_calculation = None

    def rasterize(self, subset=None, execute=False):
        """
        Write the SIMPSON input files for the provided
        :class:`morty.atomistic.CellModeller`.

        Parameters
        ----------
        subset : tuple of ints
            If provided, only the calculations in the range
            (from, to) are created
        execute : bool
            If :const:`True`, try to run the simulation locally with
            the program provided in `SimpsonCaller.simpson_executable`.

        """
        if self.cell_modeller is None:
            raise RuntimeError('No CellModeller supplied.')

        if self.results is None:
            self.results = [None] * len(self.cell_modeller)

        if subset is None:
            subset = (0, len(self.cell_modeller))
        for i in range(*subset):
            if self.results[i] is None:
                spinsystem = self.spinsystem_creator.create_spinsystem(
                    self.cell_modeller[i])
                simpson_calculation = SimpsonCalculation(
                    input_file=str(i) + '-' + self.sim_file,
                    output_file=str(i) + '-' + self.sim_file[:-3] + self.result_extension,
                    folder=self.calc_dir,
                    body=self._get_final_body(spinsystem),
                    executable=self.simpson_executable)
                if execute is True:
                    simpson_calculation.run()
                    self.results[i] = simpson_calculation.read()
        return self.results

    def load_bodyfile(self, filename):
        """
        Wrapper to quickly load a body file template.

        Parameters
        ----------
        filename : str
            Filename of the template.

        """
        with open(filename, 'r') as file:
            self.body_template = file.read()

    def write_file(self):
        """
        Write SIMPSON input files for a provided
        :class:`morty.atomistic.Spinsystem`.

        """
        if self.spinsystem is None:
            raise RuntimeError('No Spinsystem set.')
        self.simpson_calculation = SimpsonCalculation(
            input_file=self.sim_file,
            output_file=self.sim_file[:-3] + self.result_extension,
            folder=self.calc_dir,
            body=self._get_final_body(),
            executable=self.simpson_executable)
        self.simpson_calculation.write_file()

    def run(self):
        """
        Run a SIMPSON calculation locally for a provided
        :class:`morty.atomistic.Spinsystem` and read the
        results.

        """
        self.write_file()
        self.simpson_calculation.run()
        self.results = self.simpson_calculation.read()

    def get_result(self):
        """
        Returns the result of the SIMPSON calculation.

        """
        if self.cell_modeller is not None:
            if self.results is None:
                self.results = [None] * self.cell_modeller.total_nos
            for i in range(len(self.cell_modeller)):
                self.results[i] = np.loadtxt(os.path.join(
                    self.calc_dir, str(i) + '-' + self.sim_file[:-3] + self.result_extension))
            return self.results

        self.results = np.loadtxt(os.path.join(
            self.calc_dir, self.sim_file[:-3] + self.result_extension))
        return self.results

    def _get_simpson_spinsys(self, spinsystem=None):
        spinsystem = self.spinsystem if spinsystem is None else spinsystem
        channels = []
        nuclei = []
        for nucleus in spinsystem.nuclei:
            if str(nucleus.mass) + str(nucleus.atom_type) not in channels:
                channels.append(str(nucleus.mass) +
                                str(nucleus.atom_type))
            nuclei.append(str(nucleus.mass) + str(nucleus.atom_type))
        output = 'channels ' + ' '.join(channels) + '\n'
        output += 'nuclei ' + ' '.join(nuclei) + '\n'
        if self.include_csa is True:
            i = 1
            for nucleus in spinsystem.nuclei:
                output += ('    shift ' + str(i) + ' ' +
                           str(nucleus.properties['csatensor'].hms[0]) + 'p ' +
                           str(nucleus.properties['csatensor'].hms[1]) + 'p ' +
                           str(nucleus.properties['csatensor'].hms[2]) + ' ' +
                           ' '.join(str(np.degrees(x)) for x in
                                    nucleus.properties[
                                        'csatensor'].euler_pas_to_cas) + '\n')
                i += 1
        if self.include_quadrupolar is True:
            i = 1
            for nucleus in spinsystem.nuclei:
                try:
                    output += (
                        '    quadrupole ' + str(i) + ' 2 ' +
                        str(nucleus.properties['efgtensor'].cq
                           ) + " " +
                        str(nucleus.properties['efgtensor'].eta
                           ) + " " + " ".join([str(x) if x is not None
                                               else str(0)
                                               for x in
                                               np.degrees(nucleus.properties['efgtensor'].
                                                          euler_pas_to_cas)]) + "\n")
                except KeyError:
                    warnings.warn("No quadrupolar Tensor for " +
                                  nucleus.atom_type)
                i += 1
        if self.include_dipole is True:
            for dipole in spinsystem.dd_couplings:
                if self.include_dipole_euler is True:
                    euler = ' '.join(str(np.degrees(x)) for x in
                                     dipole[2].euler_pas_to_cas)
                else:
                    euler = ' 0 0 0'
                # dipole[0] & dipole[1] : atoms
                # dipole[2]: dipole coupling tensor
                # dipole[3]: dipole coupling scaling factor
                output += ('dipole ' + ' '.join(str(x + 1) for x
                                                in dipole[0:2]) + ' ' +
                           str(dipole[2].coupling_constant / (2 * math.pi) *
                               dipole[3]) + ' ' + euler + '\n')
        return output

    def _get_final_body(self, spinsystem=None):
        spinsystem = self.spinsystem if spinsystem is None else spinsystem
        replaces = [('[[spinsystem]]', self._get_simpson_spinsys(spinsystem))]
        out = self.body_template
        for replace in replaces:
            out = out.replace(replace[0], replace[1])
        return out


    def optimize(self, start_values, experimental_data, simulation_processor=None, **opt_keywords):
        """
        Run an optimization.

        Parameters
        ----------
        **opt_keywords
            All keyword arguments are passed to lmfits lbfgsb method.
        Returns
        -------
        result : tuple (array of optimized parameters, RMSD)
            The result of the optimization.

        Notes
        -----
        Use with epsilon=1e-3, or a value suiting your needs, depending on what
        you are optimizing. The optimizer will fail, if the initial step size
        is so small that there is no change in the result of a simpson
        simulation. That means, if you are optimizing a distance, make sure,
        that it changes by at least 1e-3 Å.

        """
        parameters = lmfit.Parameters()
        for i in range(len(self.cell_modeller.optimizer_boundaries())):
            parameters.add('transformation' + str(i), value=start_values[i],
                           min=self.cell_modeller.optimizer_boundaries()[i][0],
                           max=self.cell_modeller.optimizer_boundaries()[i][1])

        def rmsd(args):
            """
            Helper function used by the optimizer that returns the deviation
            between simulation and experiment.

            """
            args = list(args.valuesdict().values())
            spinsystem = self.spinsystem_creator.create_spinsystem(
                self.cell_modeller.optimizer_interface(args))
            simpson_calculation = SimpsonCalculation(
                input_file=self.sim_file,
                output_file=self.sim_file[:-3] + self.result_extension,
                folder=self.calc_dir,
                body=self._get_final_body(spinsystem),
                executable=self.simpson_executable)
            simpson_calculation.run()
            simulation = simpson_calculation.read()
            if simulation_processor is not None:
                simulation = simulation_processor(simulation)
            print(simulation)
            return simulation - experimental_data

        minimizer = lmfit.minimize(rmsd, parameters, **opt_keywords)
        results, uncert = ([None] * len(self.cell_modeller.optimizer_boundaries()),
                           [None] * len(self.cell_modeller.optimizer_boundaries()))
        for i in range(len(self.cell_modeller.optimizer_boundaries())):
            results[i] = minimizer.params['transformation' + str(i)].value
            uncert[i] = minimizer.params['transformation' + str(i)].stderr
        return results, uncert, minimizer


class SimpsonCalculation:
    """
    A simple class that holds a SIMPSON calculation. Do not use directly.

    """
    def __init__(self, input_file, output_file, folder, body, executable=None):
        """
        Instantiates a SimpsonCalculation.

        Parameters
        ----------
        input_file : str
            Filename of the SIMPSON input.
        output_file : str
            Filename of the output.
        folder : str
            Path to the SIMPSON input.
        body : str
            Content of the SIMPSON input.
        executable : str, optional
            If provided the calculation can be run locally.

        """
        self.input_file = input_file
        self.output_file = output_file
        self.folder = folder
        self.body = body
        self.executable = executable

    def write_file(self):
        """
        Writes the simulation input to the simulation file.

        """
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        with open(os.path.join(self.folder, self.input_file), 'w') as myfile:
            myfile.write(self.body)

    def run(self, wait=True):
        """
        Performs the SIMPSON calculation locally, if `SimpsonCalculation.executable` is set.

        Parameters
        ----------
        wait : bool
            If :const:`True`, wait until the calculation is finished. Otherwise the process
            will run in the background.

        """
        self.write_file()
        process = subprocess.Popen([self.executable,
                                    self.input_file],
                                   cwd=self.folder)
        if wait is True:
            process.wait()
            return True
        return process.pid

    def read(self):
        """
        Reads the resulting output with np.loadtxt().

        """
        return np.loadtxt(os.path.join(self.folder, self.output_file))


class SpinsystemCreator:
    """
    Class, that can create a :class:`morty.atomistic.Spinsystem` out of a
    :class:`morty.atomistic.Cell` instance.

    This class holds information about how to create a
    :class:`morty.atomistic.Spinsystem` from a :class:`morty.atomistic.Cell`
    instance. It stores these information to allow for fast recurrent
    creation of spinsystems, which can be used for optimizations.

    """

    def __init__(self, atom_numbers, dd_couplings=(), onlyshortest=False,
                 dipole_scaling_factor=1):
        """
        Instantiates a SpinsystemCreator.

        Parameters
        ----------
        atomNumbers : list
            List handing over the number of the atoms to include in the spin
            system.
        dd_couplings : list of lists/tuples
            Hand over the couplings to account for. Is given in indices of the
            *atomNumbers* array. Let us assume you defined *atomNumbers* as
            [5, 9, 12] and you want to include the dipolar coupling 5-9 and
            5-12, then *dd_couplings* will be [[0,1], [0, 2]]
        onlyshortest : bool
            Set to true to only take the shortest distance between the atoms
            into account, even if it is to a translated of the original unit
            cell. If False, the distance will be measured just within the
            original unit cell, which is to say within the coordinates as given
            in the :class:`morty.atomistic.Cell` instance.

        """
        self.static_atoms = tuple(atom_numbers)
        self.static_dd = [list(x) + [dipole_scaling_factor] for x in
                          dd_couplings]
        self.atoms_in_range = []
        self.onlyshortest = onlyshortest

    def add_atoms_in_range(self, center_atom, distance, atoms,
                           dd_with_atoms=(), dd_with_each_other=False,
                           max_num_spins=None, dipole_scaling_factor=1):
        """
        Dynamically includes atoms within a range around a given atom.

        Notes
        -----
        The center atom IS NOT automatically included.

        Parameters
        ----------
        center_atom : int
            Center atom from which the distance is measured.
        distance : float
            Maximum distance (in m) from `center_atom` within which to include
            new atoms.
        atoms : array of int
            The numbers of the atoms for which to calculate the distance and
            ultimately to add to the spinsystem.
        dd_with_atoms : array
            Array with atom ids (same order as upon initialization) of static
            atoms whose DD coupling to 'center_atom' should be included.
        dd_with_each_other : bool
            Set to *True* to include the dd coupling of all atoms in the added
            set with each other.
        max_num_spins : int
            Caps the maximum number of spins added to the spin system created.
            I.e. assume you have set *max_num_spins* to 2. You now add
            atoms via *add_atoms_in_range()*, say this would result in 5
            additional spins, you will end up with a spin system of 3 atoms
            total, i.e. 2 additional spins, nevertheless.
        dipole_scaling_factor : float
            Scaling factor to be applied to the dipolar coupling constant.

        """
        self.atoms_in_range.append((center_atom, atoms, distance,
                                    dd_with_atoms, dd_with_each_other,
                                    max_num_spins, dipole_scaling_factor))

    def _get_atoms_in_range(self, mycell):
        dynamic_atoms = []
        dynamic_dd = []
        dynamic_atom_numbers = []
        for entry in self.atoms_in_range:
            # Use an extra variable here, so we can add DD between each entry.
            my_dynamic_atoms = []
            my_dynamic_atom_numbers = []

            if entry[5] is not None:
                atomdiststouse = []
                tmp = mycell.get_distances_upto(
                    [mycell.get_atom(entry[0])],
                    [mycell.get_atom(x) for x in entry[1]], entry[2])
                atomdiststouse_sorted = sorted(tmp, key=lambda y: y[2]
                                              )[0:entry[5]]
                for distance_sorted in atomdiststouse_sorted:
                    for distance in tmp:
                        if distance is distance_sorted:
                            atomdiststouse.append(distance)
            else:
                atomdiststouse = mycell.get_distances_upto(
                    [mycell.get_atom(entry[0])],
                    [mycell.get_atom(x) for x in entry[1]], entry[2])

            for in_range in atomdiststouse:
                if (in_range[1][0].position_abs ==
                        in_range[1][1]) is not True:
                    # I think we don't need deepcopy here: the CSA tensor stays
                    # untouched.
                    myatom = copy.copy(in_range[1][0])
                    myatom.position_abs = in_range[1][1]
                    atomn = mycell.atoms.index(in_range[1][0]) + 1
                else:
                    myatom = in_range[1][0]
                    atomn = mycell.atoms.index(myatom) + 1
                my_dynamic_atoms.append(myatom)
                my_dynamic_atom_numbers.append(atomn)
                # Some day you will screw things up because of the following!
                # We assume, that these dynamic atoms are given to Spinsystem()
                # in second position, so we calculate the atom id with this
                # assumtion...
                # entry[3] = dd_with_atoms
                for dipole_with in entry[3]:
                    dynamic_dd.append((dipole_with,
                                       len(self.static_atoms) +
                                       len(dynamic_atoms) +
                                       len(my_dynamic_atoms) - 1,
                                       entry[6]))
            # entry[4] = dd_with_each_other
            if entry[4] is True:
                for i in range(len(my_dynamic_atoms)):
                    for j in range(i + 1, len(my_dynamic_atoms)):
                        dynamic_dd.append((i + len(self.static_atoms) +
                                           len(dynamic_atoms),
                                           j + len(self.static_atoms) +
                                           len(dynamic_atoms),
                                           entry[6]))
            dynamic_atoms += my_dynamic_atoms
            dynamic_atom_numbers += my_dynamic_atom_numbers
        return dynamic_atoms, dynamic_dd, dynamic_atom_numbers

    def create_spinsystem(self, cell):
        """
        Creates a :class:`morty.atomistic.Spinsystem` from a given
        :class:`morty.atomistic.Cell`.

        Parameters
        ----------
        cell : :class:`morty.atomistic.Cell`

        Returns
        -------
        spinsystem : :class:`morty.atomistic.Spinsystem`

        """
        dynamic_atoms, dynamic_dd, dynamic_atom_numbers = \
            self._get_atoms_in_range(cell)
        my_atom_numbers = self.static_atoms + tuple(dynamic_atom_numbers)
        if self.onlyshortest is False:
            return Spinsystem([cell.get_atom(nucleus)
                               for nucleus in self.static_atoms] +
                              dynamic_atoms, self.static_dd + dynamic_dd,
                              my_atom_numbers)
        nucs = [cell.get_atom(self.static_atoms[0])]
        nucs.extend([cell.get_shortest_distance(
            cell.get_atom(self.static_atoms[0]),
            cell.get_atom(otheratom),
            returnnewatom2=True)[1]
                     for otheratom in self.static_atoms[1:]])
        return Spinsystem(nucs + dynamic_atoms,
                          self.static_dd + dynamic_dd, my_atom_numbers)
