"""
Class to handle large amount of structures, e.g. created by an MD
simulation.

"""
import os
import warnings
import copy
import numpy as np
import scipy

from .atom import Atom
from .cell import Cell


__all__ = ['Trajectory']


class Trajectory:
    """
    Class to handle trajectories of cells.

    Trajectories can be read in, edited, and evaluated w.r.t. structural,
    energetic and spectroscopic attributes. A Trajectory can be read in from
    suitable GAUSSIAN or CASTEP files as well as from a folder structure.

    """

    def __init__(self, filename=None, atomnos=None, start=1, stop=None,
                 step=1):
        """
        Class to handle trajectories of cells.

        Parameters
        ----------
        filename : str, optional
            File to read the trajectory from. The type is determined by the
            extension.
        atomnos : array
            If given, only these atoms will be loaded. Starts with 1. Only
            implemented for CASTEP for now.
        start : int
            If given, starting step.
        stop : int
            If given, last step to read.
        step : int
            If given, only read every n-th step.

        """
        self.steps = list()

        if filename is not None:
            if filename.endswith('.md'):
                self.load_castep_trajectory(filename, task='md',
                                            atomnos=atomnos, start=start,
                                            stop=stop, step=step)
            elif filename.endswith('.geom'):
                self.load_castep_trajectory(filename, task='geom',
                                            atomnos=atomnos, start=start,
                                            stop=stop, step=step)
            elif filename.endswith('.log'):
                self.load_gaussian_trajectory(filename)
            else:
                warnings.warn('File not recognized.', RuntimeWarning)

    def __iter__(self):
        for step in self.steps:
            yield step

    def __getitem__(self, item):
        return self.steps[item]

    def __len__(self):
        return len(self.steps)

    def append_step(self, step):
        """
        Append a step to a trajectory.

        Parameters
        ----------
        step : Cell() instance


        """
        self.steps.append(step)

    def evaluate_angles(self, atomnumber1, atomnumber2, atomnumber3):
        """
        Hands back an array holding the angles between three atoms over the
        trajectory. Second atom is in the middle.

        Parameters
        ----------
        atomnumber1, atomnumber2, atomnumber3 : int
            Number of the atoms (starting at 1) spanning the angle.
        """
        angles = np.zeros(len(self.steps))
        for i in range(len(self.steps)):
            angles[i] = self.steps[i].get_angle(
                atomnumber1,
                atomnumber2,
                atomnumber3)
        return angles

    def evaluate_dihedrals(self, atomnumber1, atomnumber2, atomnumber3,
                           atomnumber4):
        """
        Hands back an array holding the dihedral angles between three atoms
        over the trajectory. Second and third atoms span the middle vector.

        Parameters
        ----------
        atomnumber1, atomnumber2, atomnumber3, atomnumber4 : int
            Number of the atoms (*starting at 1*) spanning the angle.
        """
        dihedrals = np.zeros(len(self.steps))
        for i in range(len(self.steps)):
            dihedrals[i] = self.steps[i].get_dihedral(
                self.steps[i].get_atom(atomnumber1),
                self.steps[i].get_atom(atomnumber2),
                self.steps[i].get_atom(atomnumber3),
                self.steps[i].get_atom(atomnumber4))
        return dihedrals

    def evaluate_distances(self, atomnumber1, atomnumber2):
        """
        Hands back an array holding the distances of two atoms over the
        trajectory.

        Parameters
        ----------
        atomnumber1, atomnumber2 : int
            Number of the atoms defining the distance.
        """
        distances = np.empty(len(self.steps))
        for i in range(len(self.steps)):
            distances[i] = self.steps[i].get_shortest_distance(
                self.steps[i].get_atom(atomnumber1),
                self.steps[i].get_atom(atomnumber2))[0]
        return distances

    def evaluate_energies_kjpmol(self):
        """
        Hands back an array holding the energies for each step of the
        trajectory.

        Parameters
        ----------
        atomnumber1, atomnumber2, atomnumber3, atomnumber4 : int
            Number of the atoms (starting at 1) spanning the angle.
        """
        energies = np.zeros(len(self.steps))
        for i in range(len(self.steps)):
            energies[i] = (self.steps[i].properties['totalenergy'] *
                           scipy.constants.N_A)
        return energies

    def load_castep_energy(self, ignore_warning=False, ignore_finished=False):
        """
        Load CASTEP energy info for each step in the trajectory.

        This function will load an CASTEP castep file for each step, provided
        the trajectory is generated from a folder structure, and not loaded
        from a single trajectory file. The foldernames are read from the
        respective step/cell within the trajectory.

        Parameters
        ----------

        Returns
        -------
        not_found : list
            List with the indices of steps for which no energy could be
            read.

        Notes
        -----
        This function is naturally useless if the trajectory was read from a
        '.geom' or '.md' file, as in that case the energies are contained in
        the latter files.

        """
        not_found = []
        mywarnings = []
        for step in range(len(self.steps)):
            try:
                mywarnings += [self.steps[step].load_castep_energy(
                    ignore_warning=ignore_warning,
                    ignore_finished=ignore_finished)]
            except FileNotFoundError:
                not_found.append([self.steps[step].cellname])
        return not_found, mywarnings

    def load_castep_charges(self):
        """
        Load CASTEP charges for each step in the trajectory.

        This function will load an CASTEP magres file for each step, provided
        the trajectory is generated from a folder structure, and not loaded
        from a single trajectory file. The foldernames are read from the
        respective step/cell within the trajectory.

        Returns
        -------
        not_found : list
            List with the indices of steps for which no energy could be
            read.

        Notes
        -----
        This function is naturally useless if the trajectory was read from a
        '.geom' or '.md' file, as in that case the energies are contained in
        the latter files. Be aware that these files do not list dispersion
        corrected energies, as far as I am aware.
        """
        not_found = []
        for step in range(len(self.steps)):
            try:
                self.steps[step].load_castep_energy()
            except ValueError:
                not_found.append(step)
        return not_found

    def load_castep_magres(self):
        """
        Load CASTEP magres info for each step in the trajectory.

        This function will load an CASTEP magres file for each step, provided
        the trajectory is generated from a folder structure, and not loaded
        from a single trajectory file. The foldernames are read from the
        respective step/cell within the trajectory.

        Returns
        -------
        not_found : list
            List with the indices of steps for which no magres file could be
            read.

        """
        not_found = []
        for step in range(len(self.steps)):
            try:
                self.steps[step].load_castep_magres()
            except FileNotFoundError:
                not_found.append(step)
        return not_found

    def load_castep_trajectory(self, trajfilename, task='geom', atomnos=None,
                               start=1, stop=None, step=1):
        """
        Loads a CASTEP trajectory, either from a geometry optimisation or a
        molecular dynamics simulation.

        Parameters
        ----------
        trajfilename : str
            Filename of the trajectory file.
        task : str
            Either 'geom' (default) or 'md', for geometry optimizations or MD
            simulations, respectively.
        atomnos : array
            If given, only these atoms will be loaded. Starts with 1.

        """
        file = open(trajfilename, 'r')
        headermarker = 0
        current_cell = list()
        stepno = 0

        for line in file:
            line = line.replace("\x00"," ").strip()
            if 'begin header' in line.lower():
                headermarker = 1
                continue
            elif 'end header' in line.lower():
                headermarker = 2
                continue
            elif headermarker == 1 or not line:
                continue

            else:
                myline = line.split('<--')
                if len(myline) == 1 or myline[1] == ' c': # <-- c is CASTEP 17 style
                    current_cell = []
                    current_atom = 1
                    stepno += 1
                    if (stepno >= start and (stepno - 1) % step == 0 and
                            (stepno < stop if stop is not None else True)):
                        self.steps.append(Cell())
                        if task == 'md':
                            self.steps[-1].properties['time'] = (
                                float(line) * scipy.constants
                                .physical_constants['atomic unit of time'][0])

                elif (stepno >= start and (stepno - 1) % step == 0 and
                      (stepno <= stop if stop is not None else True)):
                    if myline[1] == ' E':
                        line_contents = [float(x) for x in myline[0].split()]
                        self.steps[-1].properties['totalenergy'] = (
                            float(line_contents[0]) *
                            scipy.constants.physical_constants[
                                'atomic unit of energy'][0])
                        self.steps[-1].properties['hamiltonianenergy'] = (
                            float(line_contents[1]) *
                            scipy.constants.physical_constants[
                                'atomic unit of energy'][0])
                        if len(line_contents) == 3:
                            self.steps[-1].properties['kineticenergy'] = (
                                float(line_contents[2]) *
                                scipy.constants.physical_constants[
                                    'atomic unit of energy'][0])

                    elif myline[1] == ' T':
                        line_contents = [float(x) for x in myline[0].split()]
                        self.steps[-1].properties['currentTemperature'] = (
                            float(line_contents[0]) *
                            scipy.constants.physical_constants[
                                'atomic mass unit-joule relationship'][0])

                    elif myline[1] == ' P':
                        line_contents = [float(x) for x in myline[0].split()]
                        self.steps[-1].properties['pressure'] = (
                            float(line_contents[0]))

                    # Cell is given in bohr here, so we have to convert to
                    # angstrom
                    elif myline[1] == ' h':
                        line_contents = [float(x) for x in myline[0].split()]
                        current_cell.append(np.array(
                            [(x * scipy.constants.physical_constants[
                                'atomic unit of length'][0])
                             for x in line_contents]).astype(np.float32))
                        if len(current_cell) == 3:
                            self.steps[-1].lattice_cart = np.array(
                                [current_cell[0], current_cell[1],
                                 current_cell[2]])
                            cur_basis = self.steps[-1].lattice_cart
                            cur_basis_i = np.linalg.inv(cur_basis)

                    elif myline[1] == ' R':
                        if atomnos is None or current_atom in atomnos:
                            line_contents = myline[0].split()
                            # Coordinates given in bohr here, so we have to convert
                            # to angstrom
                            abs_vector = np.array([
                                float(line_contents[2]) *
                                scipy.constants.physical_constants[
                                    'atomic unit of length'][0],
                                float(line_contents[3]) *
                                scipy.constants.physical_constants[
                                    'atomic unit of length'][0],
                                float(line_contents[4]) *
                                scipy.constants.physical_constants[
                                    'atomic unit of length'][0],
                                ])
                            self.steps[-1].mod_append_atom(
                                Atom(position_abs=abs_vector,
                                     basis=cur_basis, basis_i=cur_basis_i,
                                     atom_type=line_contents[0]))
                            current_atom += 1
        file.close()

    def load_gaussian_trajectory(self, trajfilename, max_x=50*1e-10,
                                 max_y=50*1e-10, max_z=50*1e-10):
        """
        Read in a trajectory from G09.

        Parameters
        ----------
        trajfilename : str
            Filename of the trajectory file.
        max_x, max_y, max_z : float
            Size of the cell, in which a molecule is being put.

        ..note ::
        For now, only MD is supported!

        """
        file = open(trajfilename, 'r')
        found_step = False
        found_atoms = False
        found_atomtypes = False
        atomtypes_present = []

        for line in file:
            line_contents = [x for x in line.split()]
            if 'Summary information for step' in line:
                found_step = True
                self.steps.append(Cell())
                self.steps[-1].lattice_abc = [max_x, max_y, max_z,
                                              np.radians(90.0),
                                              np.radians(90.0),
                                              np.radians(90.0)]
            if 'Symbolic Z-Matrix:' in line:
                found_atomtypes = True
            elif line == '':
                found_atomtypes = False
            elif found_atomtypes is True:
                atomtypes_present.append(line_contents[0].upper())
            if ('TRJ-TRJ-TRJ-TRJ-TRJ' in line or
                    'Final analysis for' in line):
                found_step = False
            if found_step is True:
                if 'Time' in line:
                    self.steps[-1].properties['time'] = (
                        np.float(line_contents[2]) * 1e-15)
                if 'Total energy' in line:
                    self.steps[-1].properties['totalenergy'] = (
                        float(line_contents[2].replace('D', 'E')) *
                        scipy.constants.physical_constants[
                            'atomic unit of energy'][0])
                if 'Cartesian coordinates:' in line:
                    found_atoms = True
                elif found_atoms is True and 'I= ' in line:
                    # Coordinates given in bohr here, so we have to convert
                    # to angstrom
                    abs_vector = np.array([
                        (np.float(line_contents[3].replace('D', 'E')) *
                         scipy.constants.physical_constants[
                             'atomic unit of length'][0]),
                        (np.float(line_contents[5].replace('D', 'E')) *
                         scipy.constants.physical_constants[
                             'atomic unit of length'][0]),
                        (np.float(line_contents[7].replace('D', 'E')) *
                         scipy.constants.physical_constants[
                             'atomic unit of length'][0])])
                    self.steps[-1].mod_append_atom(
                        Atom(position_abs=abs_vector,
                             basis=self.steps[-1].lattice_cart,
                             atom_type=atomtypes_present[
                                 len(self.steps[-1].atoms)]))
                elif found_atoms is True and 'I= ' not in line:
                    found_atoms = False

    def mod_make_supercell(self, supercell=None):
        """
        Create a supercell.

        Parameters
        ----------
        supercell : list
            gives the number of cells to expand in ``x, y, z``.

        """
        expand = []
        for expand_x in range(supercell[0]):
            for expand_y in range(supercell[1]):
                for expand_z in range(supercell[2]):
                    expand.append([expand_x, expand_y, expand_z])
        expand.remove([0, 0, 0])
        for step in self.steps:
            atoms_to_append = []
            for new_cell in expand:
                tmp_atoms = copy.deepcopy(step.atoms)
                for myatom in tmp_atoms:
                    myatom.set_position_frac(myatom.position_frac +
                                             np.array(new_cell))
                atoms_to_append.extend(tmp_atoms)
            step.atoms.extend(atoms_to_append)
            step.update_lattice_abc(np.concatenate(
                [[step.get_lattice_abc()[i] * supercell[i] for i in range(3)],
                 step.get_lattice_abc()[3:6]]), fix_abs=True)

    def set_up_job(self, calcfolder=None,
                   program='gaussian', write_queue_file=False,
                   queue_template=None,
                   update_submitall=None):
        """
        Sets up a job for the cell within the cell.

        Provides an interface to the `jobs` module. Allows the user to set up
        a G09 or CASTEP caclulation directly from a `Cell()` instance.
        All necessary files are automatically created in the appropriate
        folders, ready to be handled by local or queued execution.
        The function can also be used to just 'update' the job, for example
        to rewrite the queue submission file without touching other files, if
        you wish to run the job on another machine.

        Parameters are essentially the same as for `jobs.Job()`.

        Parameters
        ----------
        threads : int
            Number of threads to use for the job. CASTEP and G09 are set up
            automatically to use mpi or the internal threading mechanism,
            respectively. The peculiarities of the respective queuing systems
            are accounted for.
        calcfolder : string, optional
            The folder in which the calculations are placed within.
            The individual folders are then placed within the folder and named
            after the cellname of the respective cell/step.
        write_queue_file : tuple of bool
            The first bool triggers the write out the files for the respective
            program to use. For G09, this is a ''.com'' file, for CASTEP a
            ''.cell'' and a ''.param'' file.
            The second bool triggers the write out the queueing files for the
            job created from the cell.
        reuse_geomopt : bool
            If True, will first extract the last step from a geometry
            optimisation. For now, only accepts CASTEP runs. Relies on a
            *.geom* file to be present in the calculation folder.

        Returns
        -------

        Notes
        -----
        jobname, jobname_q
            The jobname is taken from the cell instance's cellname.

        """
        updated_steps = []
        submit_all_string = str()
        if os.path.isdir(calcfolder) is not True:
            os.mkdir(calcfolder)
        for step in self.steps:
            stepcalcfolder = os.path.join(calcfolder, step.cellname)
            step.set_up_job(
                calcfolder=stepcalcfolder,
                jobname=step.cellname,
                program=program,
                write_queue_file=write_queue_file,
                queue_template=queue_template)
            if update_submitall is None:
                submit_all_string += str("cd " + stepcalcfolder + "; qsub " +
                                         step.cellname +
                                         "_qsub.sh; cd ..;\n")
                updated_steps.append(step.cellname)
            elif (update_submitall[0](step[-1],
                                      *(update_submitall[1])) is
                  update_submitall[2]):
                submit_all_string += str("cd " + stepcalcfolder + "; qsub " +
                                         step.cellname +
                                         "_qsub.sh; cd ..;\n")
                updated_steps.append(step.cellname)
        with open(os.path.join(calcfolder, 'submitusall.sh'), 'w') as \
                submitall_file:
            submitall_file.write(submit_all_string)
