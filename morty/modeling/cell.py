"""
Cell class that can load different file formats and supports
various routines for modeling.

"""

import copy
import math
import re
import os
import warnings
import shlex
import numpy as np
from numpy.linalg import norm
import scipy.constants

from .. import constants
from ..util import fileio
from ..util import axis_rotation_matrix
from .atom import Atom
from .tensor import CSATensor, EFGTensor
from ..util import HALL_SYMBOLS, HM_SYMBOLS

try:
    import spglib._spglib as spglib
    SPGLIB_AVAIL = True
except ImportError:
    SPGLIB_AVAIL = False


__all__ = ['Cell']


class Cell:
    """
    Class to handle periodic crystallographic structure data.

    Class which can handle reading of various crystallographic
    input files, can do analysis and manipulate the structure.

    Attributes
    ----------
    castep_paramfile : dict
        Holds all parameters set in a CASTEP *param* file.
    castep_cellfile_additions : dict
        Similar to `castep_paramfile`, but holds parameters which are set
        within the *cell* file. Includes the key ``additional_lines``,
        which is sort of the garbage collector for parameters not
        implemented.
    cellname : str
        Name of the cell, e.g when writing out files or jobs.
    constraints : dict
        Constraints set for the cell, not very well implemented.
    foldername : str
        The folder to be used to write out files automatically.
    gaussian_params : dict
        Provides parameters for G09 calcs.
    properties : dict
        A dictionary containing properties of the cell.

    Notes
    -----
    Iterating over this class will yield all the atoms contained in this cell.
    Its length is equal to the number of atoms in the cell.

    Some common `properties` keys are:

        - totalenergy : float
            The total energy of the structure.
        - shielding_by_group : dict
            The shieldings of the atoms in the cell, sorted by the
            chemical group. See
            :class:`morty.atomistic.Cell.det_shifts_by_group()`
            for details.

    The `gaussian_params` dictionary holds the following keys:

        - filename : string
            Full path to the output file. Defaults to 'myinput.com'.
        - method : string
            Method to use. Defaults to 'pbe1pbe'.
        - basis_set : string
            The basis set to use. Defaults to '6-31G'.
        - remaining_route_section : string
            The route section telling GAUSSIAN what to do, sparing
            method and basis set.
        - charge_and_multiplicity : string
            Hands over spin and multiplicity for the given geometry.
            Defaults to ``0 1``.
        - title : string
            The title used within G09. Defaults to 'fancy_default_title'.
        - link0_section : list of strings
            The link0 section of the GAUSSIAN input. Contains stuff like
            ``%NProcShared``. Each element contains one line.
            Defaults to ``None``.
        - multi_step_route_section : list of strings
            The route sections for subsequent jobs using the same
            wavefunction, i.e. the checkfile of, the preceding calculation.
            Each element in the list will result in one calculation.
            The checkfile required is automatically written.
            Defaults to None.
        - multi_step_titles : list of strings
            Titles of the multi step jobs to follow the first job.
            Defaults to ``None``.
        - write_checkfile : int
            If to retain a checkfile. May be 0, 1, 2, 3.
            | 0: Do not save a checkfile whatsoever.
            | 1: Save a checkfile.
            | 2: Save a formatted checkfile.
            | 3: Save a formatted checkfile as well as a checkfile.
            Defaults to ``0``.

    """
    def __init__(self, filename=None):
        """
        Class to handle periodic crystallographic structure data.

        Parameters
        ----------
        filename : str
            File to read a structure from. The format of the file is determined
            by the file extension.

        """
        self.properties = {}
        #: Job instance if needed.
        self.cellname = None
        self.foldername = '.'
        self.constraints = {}
        self.castep_paramfile = {}
        self.castep_cellfile_additions = {}
        self.castep_cellfile_additions['additional_lines'] = []
        self.gaussian_params = {}

        self.lattice_cart = None
        self.atoms = []
        self.spacegroup_hallnum = 1
        self.spacegroup_hallsymbol = 'P 1'
        self._custom_mod_functions = []
        self.nmr_reference = None
        self.symmetry_operations_applied = False

        if filename is not None:
            if filename.endswith('.cell'):
                self.load_castep_cellfile(filename)
            elif filename.endswith('.log'):
                self.load_gaussian_logfile(filename)
            elif filename.endswith('.cif'):
                self.load_cif(filename)
            elif filename.endswith('.magres'):
                self.load_castep8_magres(filename)
            elif filename.endswith('.xyz'):
                self.load_xyzfile(filename)
            else:
                warnings.warn('File not recognized.', RuntimeWarning)

    def __iter__(self):
        for myatom in self.atoms:
            yield myatom

    def __len__(self):
        return len(self.atoms)

    def change_isotopes(self, to_replace, replace_by):
        """
        Change isotopes of all atoms.

        This will change the isotope of all atoms in this cell. This allows,
        for example, an easy way to deuterate the cell.

        Example
        -------
        Change from proton to deuteron: ::
            mycell.change_isotopes('1H', '2H')

        Parameters
        ----------
        to_replace : str
            The isotopes to replace, e.g. '1H'.
        replace_by : str
            The new isotope, e.g. '2H'.

        """
        for myatom in self:
            if str(myatom.mass) + str(myatom.atom_type) == to_replace:
                myatom.mass = ''.join(re.findall("[0-9]+", replace_by))
                myatom.atom_type = ''.join(re.findall("[a-zA-Z]+", replace_by))

    def check_vdw_overlap(self, atomno1, otheratomnos, ratio=1,
                          verbose=False):
        """
        Checks if the shortest distance between a group of atoms and other
        atoms in the cell is below a defined portion of the VdW radii of the
        participating atoms.

        Parameters
        ----------
        atomno1 : list of int
            Numbers of the atoms to check the distance from.
        otheratomnos : list of int
            Numbers of the other atoms to check the distance against.
        ratio : float
            The fraction of the sum of the VdW radii below which to return
            :const:`True`.
        verbose : bool
            If :const:`True`, return extended output containing the atoms with which an
            overlap was detected.

        Returns
        -------
        check : list of bool, (list)
            :const:`True` if one or more overlaps were detected.
        check_atoms : list of int
            List of atomnumbers, for which the overlap was detected, if verbose
            is True.

        """
        check = False
        check_atoms = []
        for atomno2 in otheratomnos:
            if (self.get_shortest_distance(self.get_atom(atomno1),
                                           self.get_atom(atomno2))[0] <
                    (self.get_atom(atomno1).get_vdw_radius() +
                     self.get_atom(atomno2).get_vdw_radius()) * ratio):
                check = True
                if verbose is True:
                    check_atoms.append(self.atoms.index(
                        self.get_atom(atomno2)) + 1)
        if verbose is False:
            return [check]
        return [check, check_atoms]

    def det_bonds(self, foldername_bondsfile=None):
        """
        Determines the bonds present in the current structure.

        All atoms have to reside inside the unit cell or this will fail.
        When a chemical bond is detected between *atom 1* and *atom 2*,
        the atoms will be recognized as first neighbours of each other
        and their `neighbours` attributes will be set accordingly.

        Parameters
        ----------
        foldername_bondsfile : string, optional
            Path to the folder, that contains the *bonds* file with the bonds
            definitions.

        Notes
        -----
        The bonds present in the structure are determined based on the
        definitions given in a file *bonds* residing in the current folder.
        A prototype of that file is found in the :class:`morty.external` folder.
        All atoms have to reside inside the unit cell or the module will fail.

        """
        if np.min([x for a in self.atoms for x in a.position_frac]) < 0:
            warnings.warn(
                'Atoms outside unitcell found. This will yield wrong results.',
                RuntimeWarning)
        for atom1 in self.atoms:
            atom1.neighbours = []
        bondsconfig = fileio.SsParser()
        if foldername_bondsfile:
            bondsconfig.read(os.path.join(foldername_bondsfile, 'bonds'))
        elif self.foldername:
            bondsconfig.read(os.path.join(self.foldername, 'bonds'))
        else:
            bondsconfig.read('bonds')

        mymax = max([float(bondsconfig.blocks['bonds'][0][key].split(':')[0])
                     for key in bondsconfig.blocks['bonds'][0].keys()])
        translations = np.array([[x, y, z] for z in [-1, 0, 1]
                                 for y in [-1, 0, 1]
                                 for x in [-1, 0, 1]]) @ self.lattice_cart
        for atom1 in self.atoms:
            mydistances = np.array([np.linalg.norm(
                atom1.position_abs - atom2.position_abs - translations, axis=1)
                                    for atom2 in self.atoms])
            indicesa2 = np.where(mydistances < mymax)
            for a2_i in range(len(indicesa2[0])):
                if atom1 != self.atoms[indicesa2[0][a2_i]]:
                    # ensure we can run the stuff twice without having
                    # neighbours doubled
                    # first check priority order, then do the distance calc
                    if (not self.atoms[indicesa2[0][a2_i]].is_neighbour(1, atom1) and
                            int(bondsconfig.blocks['bonds'][0][atom1.atom_type].split(':')[1]) >
                            int(bondsconfig.blocks['bonds'][0][
                                self.atoms[indicesa2[0][a2_i]].atom_type].split(':')[1])):
                        if (mydistances[indicesa2[0][a2_i],
                                        indicesa2[1][a2_i]] <
                                float(bondsconfig.blocks['bonds'][0][atom1.atom_type
                                                                    ].split(':')[0])):
                            atom1.set_neighbour(
                                mydistances[indicesa2[0][a2_i], indicesa2[1][a2_i]],
                                self.atoms[indicesa2[0][a2_i]],
                                int(bondsconfig.blocks['bonds'][0][self.atoms[
                                    indicesa2[0][a2_i]].atom_type].split(':')[2]),
                                translations[indicesa2[1][a2_i]] +
                                self.atoms[indicesa2[0][a2_i]].position_abs)
                            self.atoms[indicesa2[0][a2_i]].set_neighbour(
                                mydistances[indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                atom1, int(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[2]),
                                - translations[indicesa2[1][a2_i]] +
                                atom1.position_abs)
                    elif (not self.atoms[indicesa2[0][a2_i]].is_neighbour(1, atom1) and
                          int(bondsconfig.blocks['bonds'][0][
                              atom1.atom_type].split(':')[1]) <=
                          int(bondsconfig.blocks['bonds'][0][
                              self.atoms[indicesa2[0][a2_i]].atom_type].split(
                                  ':')[1])):
                        if (mydistances[indicesa2[0][a2_i], indicesa2[1][a2_i]] <
                                float(bondsconfig.blocks['bonds'][0][
                                    self.atoms[
                                        indicesa2[0][a2_i]].atom_type].split(
                                            ':')[0]) or
                                mydistances[indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]] <
                                float(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[0])):
                            atom1.set_neighbour(
                                mydistances[indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                self.atoms[indicesa2[0][a2_i]],
                                int(bondsconfig.blocks['bonds'][0][
                                    self.atoms[
                                        indicesa2[0][a2_i]].atom_type].split(':')[2]),
                                - translations[indicesa2[1][a2_i]] +
                                self.atoms[indicesa2[0][a2_i]].position_abs)
                            self.atoms[indicesa2[0][a2_i]].set_neighbour(
                                mydistances[indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                atom1,
                                int(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[2]),
                                translations[indicesa2[1][a2_i]
                                            ] + atom1.position_abs)

    def det_bonds_fast(self, foldername_bondsfile=None):
        """
        Determines the bonds present in the current structure. Large memory
        demand.

        All atoms have to reside inside the unit cell or the module will fail.
        When a chemical bond is detected between *atom 1* and *atom 2*,
        the atoms will be recognised as first neighbours of each other,
        i.e. *atom 1* will be saved as neighbour in *atom 2*'s `neighbours`
        dictionary.
        This version uses a lot of memory, don't use it with more than a few
        hundreds of atoms. It is almost 3 times faster, but since it's still
        only a few 100 ms, this is probably not of much use anymore.

        Parameters
        ----------
        foldername_bondsfile : string, optional
            Path to the folder, that contains the *bonds* file with the bonds
            definitions.

        Notes
        -----
        The bonds present in the structure are determined based on the
        definitions given in a file *bonds* residing in the current folder.
        A prototype of that file is found in the :class:`morty.external` folder.
        All atoms have to reside inside the unit cell or the module will fail.

        """
        for atom1 in self.atoms:
            atom1.neighbours = []
        if np.min([x for a in self.atoms for x in a.position_frac]) < 0:
            warnings.warn(
                'Atoms outside unitcell found. This will yield wrong results.',
                RuntimeWarning)
        bondsconfig = fileio.SsParser()
        if foldername_bondsfile:
            bondsconfig.read(os.path.join(foldername_bondsfile, 'bonds'))
        elif self.foldername:
            bondsconfig.read(os.path.join(self.foldername, 'bonds'))
        else:
            bondsconfig.read('bonds')
        coords = np.repeat(
            [np.array([[a.position_abs] * 27 for a in self.atoms])],
            len(self.atoms), axis=0)
        translation_vectors = np.repeat(
            [np.repeat([np.array([[x, y, z] for z in [-1, 0, 1]
                                  for y in [-1, 0, 1] for x in [-1, 0, 1]])],
                       len(self.atoms), axis=0)], len(self.atoms), axis=0
            ) @ self.lattice_cart
        mydistances = np.linalg.norm(
            (-coords + coords.transpose((1, 0, 2, 3)) + translation_vectors),
            axis=3)
        mymax = max([float(bondsconfig.blocks['bonds'][0][key].split(':')[0])
                     for key in bondsconfig.blocks['bonds'][0].keys()])
        for iter1, atom1 in enumerate(self.atoms):
            indicesa2 = np.where(mydistances[iter1, :] < mymax)
            for a2_i in range(0, len(indicesa2[0])):
                if atom1 != self.atoms[indicesa2[0][a2_i]]:
                    # ensure we can run the stuff twice without having
                    # neighbours doubled
                    # first check priority order, then do the distance calc
                    if (not self.atoms[indicesa2[0][a2_i]]
                            .is_neighbour(1, atom1) and
                            int(bondsconfig.blocks['bonds'][0][
                                atom1.atom_type].split(':')[1]) >
                            int(bondsconfig.blocks['bonds'][0][
                                self.atoms[indicesa2[0][a2_i]].atom_type
                                ].split(':')[1])):
                        if (mydistances[iter1, indicesa2[0][a2_i],
                                        indicesa2[1][a2_i]] <
                                float(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[0])):
                            atom1.set_neighbour(
                                mydistances[iter1, indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                self.atoms[indicesa2[0][a2_i]],
                                int(bondsconfig.blocks['bonds'][0][self.atoms[
                                    indicesa2[0][a2_i]].atom_type].split(':')[2]),
                                translation_vectors[iter1, indicesa2[0][a2_i],
                                                    indicesa2[1][a2_i]] +
                                self.atoms[indicesa2[0][a2_i]].position_abs)
                            self.atoms[indicesa2[0][a2_i]].set_neighbour(
                                mydistances[iter1, indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                atom1, int(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[2]),
                                - translation_vectors[iter1, indicesa2[0][a2_i],
                                                      indicesa2[1][a2_i]] +
                                atom1.position_abs)
                    elif (not self.atoms[indicesa2[0][a2_i]]
                          .is_neighbour(1, atom1) and
                          int(bondsconfig.blocks['bonds'][0][
                              atom1.atom_type].split(':')[1]) <=
                          int(bondsconfig.blocks['bonds'][0][
                              self.atoms[indicesa2[0][a2_i]].atom_type].split(
                                  ':')[1])):
                        if (mydistances[iter1, indicesa2[0][a2_i],
                                        indicesa2[1][a2_i]] <
                                float(bondsconfig.blocks['bonds'][0][
                                    self.atoms[
                                        indicesa2[0][a2_i]].atom_type].split(
                                            ':')[0]) or
                                mydistances[
                                    iter1, indicesa2[0][a2_i],
                                    indicesa2[1][a2_i]] <
                                float(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[0])):
                            atom1.set_neighbour(
                                mydistances[iter1, indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                self.atoms[indicesa2[0][a2_i]],
                                int(bondsconfig.blocks['bonds'][0][
                                    self.atoms[
                                        indicesa2[0][a2_i]].atom_type].split(
                                            ':')[2]),
                                - translation_vectors[iter1, indicesa2[0][a2_i],
                                                      indicesa2[1][a2_i]] +
                                self.atoms[
                                    indicesa2[0][a2_i]].position_abs)
                            self.atoms[indicesa2[0][a2_i]].set_neighbour(
                                mydistances[iter1, indicesa2[0][a2_i],
                                            indicesa2[1][a2_i]],
                                atom1,
                                int(bondsconfig.blocks['bonds'][0][
                                    atom1.atom_type].split(':')[2]),
                                translation_vectors[
                                    iter1, indicesa2[0][a2_i], indicesa2[1][a2_i]
                                    ] + atom1.position_abs)

    def det_groups(self, atom_types=None, foldername=None,
                   filename='groups', verbose=False,
                   ignore_atom_type=None, ignore_chemgroup=None):
        """
        Determines the chemical group for all atoms in the cell.
        Uses the :class:`morty.atomistic.Atom.get_group()` method.

        Parameters
        ----------
        atom_types : array_like, optional
            If set, only specific atoms are evaluated (e.g. 'C').
        verbose : bool, optional
            If :const:`True`, a warning will be emitted when a group of
            a atom couldn't be determined.
        foldername : str, optional
            If set, use this folder to search for the *groups* file. Otherwise
            the `Cell`s `foldername` (if set) or the current working directory
            is used.
        filename : str, optional
            If set, use this filename, otherwise 'groups'.
        ignore_atom_type : array_like, optional
            Ignore a certain `atom_type`. This comes in handy in models with
            adsorbed atoms you want to disregard.
        ignore_chemgroup : array_like, optional
            Ignore a certain chemical group. This comes in handy in models with
            adsorbed molecules.
            WARNING: This does NOT account for a recursive problem, where a
            first chemical group ignores a second chemical group which has in
            turn to ignore the first one to be identified.

        Notes
        -----
        Bonds, and thereby neighbours, have to be determined first,
        which can be done using the :class:`morty.atomistic.Cell.det_bonds()`.
        The groups present in the structure are determined based on the
        definitions given in a file *groups* residing in the current folder.
        A prototype of that file is found in the :class:`morty.external` folder.


        A more involved syntax allows to define patterns based on already
        known chemical groups. This is rather involved - think real good
        before you use this. ::

            %NUCLEUS
            runningnumber = 0
            type = C
            neighbours = N$NH&blue=ring_*,N,N_NH:C,C,C
            priority = 1
            name = C_np
            mustbe = 1
            %END_NUCLEUS

        Shells are delimited by ``:``, neighbouring atoms by ``,``.
        Our example features two carbons and one hydrogen in the first
        shell of neighbours and four nitrogen in the second.
        Other identifiers are supported, i.e. ``$`` to define the
        `chem_group` of the neighbour, ``&`` to request one of the
        `chem_group_additional` of the neighbour, ``=`` to request one of the
        `chem_group_additional` to be shared for the atom to be identified
        and the neighbour. For these shared `chem_group_additional`,
        wildcards are supported, e.g. ``[...]=ring_*[...]`` will get a match if
        both the atom and the respective neighbour have a ``ring_14`` as
        `chem_group_additional`, but will not if it were ``ring_13`` and
        ``ring_14``.
        The chemical group for this nitrogen must be
        known before running `det_groups`. This can for example be achieved
        by using the switch `atom_types`.

        """
        nomatch = []
        if not foldername:
            if self.foldername:
                foldername = self.foldername
            else:
                foldername = './'
        if atom_types:
            atoms = [x for x in self.atoms if x.atom_type in atom_types]
        else:
            atoms = self.atoms
        for myatom in atoms:
            if not myatom.get_group(foldername, filename,
                                    ignore_atom_type,
                                    ignore_chemgroup):
                nomatch.append(str(myatom.atom_type +
                                   str(self.atoms.index(myatom) + 1)))
        if (verbose is True) and (nomatch != []):
            nomatch += [''] * ((int(len(nomatch) / 5) + 1) * 5 - len(nomatch))
            pnm = "  " + "\n  ".join(
                ['{0:10s}{1:10s}{2:10s}{3:10s}{4:10s}'.format(
                    *nomatch[i:i + 5])
                 for i in range(0, len(nomatch), 5)])
            warnings.warn("Found no match for \n" + pnm, RuntimeWarning)

    def find_symmetry(self, precision=1e-3, angle_tolerance=-1):
        """
        Find symmetry in crystal.

        Searches for symmetry within the given precision and returns the
        spacegroup, symmetry operations and some additional information.

        Parameters
        ----------
        precision : float
            From spglib documentation:
            Tolerance of distance between atomic positions and between lengths
            of lattice vectors to be tolerated in the symmetry finding. The
            angle distortion between lattice vectors is converted to a length
            and compared with this distance tolerance. If the explicit angle
            tolerance is expected, see angle_tolerance.
        angle_tolerance : float
            Tolerance of angle between lattice vectors in degrees to be
            tolerated in the symmetry finding.

        Returns
        -------
        dataset : dict
            Dictionary with these keys:
            'number', 'hall_number', 'international', 'hall',
            'transformation_matrix', 'origin_shift', 'rotations',
            'translations', 'wyckoffs', 'equivalent_atoms', 'std_lattice',
            'std_types', 'std_positions', 'pointgroup_number', 'pointgroup'
            See spglib documentation for get_dataset() for details.

        Notes
        -----
        This function requires spglib. If it is not installed, a RuntimeError
        will be raised.

        """
        if SPGLIB_AVAIL is False:
            raise RuntimeError('Spglib not available!')

        positions = np.array([atom.position_frac for atom in self.atoms],
                             dtype='double', order='C')
        lattice = np.array(self.lattice_cart * 1e10, dtype='double', order='C')
        numbers = np.array([atom.get_nuc_charge() for atom in self.atoms],
                           dtype='intc')

        keys = ('number', 'hall_number', 'international', 'hall',
                'transformation_matrix', 'origin_shift', 'rotations',
                'translations', 'wyckoffs', 'equivalent_atoms', 'std_lattice',
                'std_types', 'std_positions', 'pointgroup_number',
                'pointgroup')
        dataset = {}
        for key, data in zip(keys, spglib.dataset(lattice, positions, numbers, 0,
                                                  precision, angle_tolerance)):
            dataset[key] = data
        return dataset

    def det_shifts_by_group(self, verbose=False):
        """
        Stores the shifts, sorted by group.

        The shieldings of all atoms are sorted by group and stored in
        self.properties['shielding_by_group']. This sorting can be used to
        plot computed shifts and color them by group.
        The chemical groups have to have been determined beforehand using
        :class:`morty.atomistic.Cell.det_bonds()` and
        :class:`morty.atomistic.Cell.det_groups()`.

        Parameters
        ----------
        verbose: bool
            If :const:`True`, prints the shifts sorted by the chemical group
            assigned to the respective atom.

        """
        self.properties['shifts_by_group'] = {}
        for myatom in self.atoms:
            try:
                if (myatom.atom_type not in
                        self.properties['shifts_by_group'].keys()):
                    self.properties['shifts_by_group'][myatom.atom_type] = {}
                if (myatom.properties['chem_group'] not in
                        self.properties['shifts_by_group'][
                            myatom.atom_type].keys()):
                    self.properties['shifts_by_group'][myatom.atom_type][
                        myatom.properties['chem_group']] = [
                            myatom.properties['csatensor'].hms[0]]
                else:
                    self.properties['shifts_by_group'][myatom.atom_type][
                        myatom.properties['chem_group']].append(
                            myatom.properties['csatensor'].hms[0])
            except:
                pass
        if verbose is True:
            print("\n")
            for mytype in self.properties['shifts_by_group'].keys():
                for mykey in self.properties['shifts_by_group'][
                        mytype].keys():
                    print("Shifts for " + mykey + ":\n" +
                          '-' * (len(mykey) + 12))
                    toprint = [str(round(v, 2)) for v in
                               self.properties['shifts_by_group'][
                                   myatom.atom_type][mykey]]
                    toprint += [''] * ((int(len(toprint) / 5) + 1) * 5 -
                                       len(toprint))
                    print("    " + "\n    ".join(
                        ['{0:10s}{1:10s}{2:10s}{3:10s}{4:10s}'.format(
                            *toprint[i:i + 5])
                         for i in range(0, len(toprint), 5)]) + "\n")

    @staticmethod
    def get_angle(atom1, atom2, atom3):
        """
        Calculate the angle spanned by three atoms.
        `atom2` is the one in the middle.

        Parameters
        ----------
        atom1, atom2, atom3 : :class:`morty.atomistic.Atom`
            Atoms between which the angle will be calculated.

        Returns
        -------
        angle : float
            The angle spanned, in radians.

        """
        vec1 = atom2.position_abs - atom1.position_abs
        vec2 = atom2.position_abs - atom3.position_abs
        return np.arccos(vec1 @ vec2 / (norm(vec1) * norm(vec2)))

    @staticmethod
    def get_angle_axes(atom1, atom2, atom3, atom4):
        """
        Calculate the angle spanned by two axes, defined by 4 atoms.
        First axis is defined by `atom1` and `atom2`, second by `atom3` and
        `atom4`.

        Parameters
        ----------
        atom1, atom2, atom3, atom4 : :class:`morty.atomistic.Atom`
            The four atoms spanning the two axes.

        Returns
        -------
        angle : float
            The angle spanned, in rad.

        """
        vec1 = atom2.position_abs - atom1.position_abs
        vec2 = atom4.position_abs - atom3.position_abs
        return np.arccos(vec1 @ vec2 / (norm(vec1) * norm(vec2)))

    def get_atom(self, name, relative_numbering=False):
        """
        Returns a certain atom.

        Returns the atom identified by `name`.

        Parameters
        ----------
        name: string or int
            Usually the absolute numbering of the atom in the order as in the
            input file. If relative_numbering is :const:`True`, the relative numbering
            is used. E.g. 'C4' is the fourth carbon atom in the order as in
            the input file and not the fourth atom.
        relative_numbering : bool, optional
            If to use relative numbering. See `name`.

        Returns
        -------
        atom : :class:`morty.atomistic.Atom`
            The atom instance requested.

        """
        if relative_numbering is True:
            counter = 0
            for myatom in self.atoms:
                if myatom.atom_type == ''.join(re.findall("[a-zA-Z]+", name)):
                    counter += 1
                    if counter == int(''.join(filter(lambda x: x.isdigit(),
                                                     str(name)))):
                        return myatom
            return None

        return self.atoms[int(''.join(filter(
            lambda x: x.isdigit(), str(name)))) - 1]

    def get_atomnumbers_within(self, centeratomnr=None,
                               centerpoint_frac=None, centerpoint_abs=None,
                               max_distance=None, min_distance=0,
                               max_distance_xyz=None,
                               min_distance_xyz=(0.0, 0.0, 0.0),
                               atom_types=None):
        """
        Allows to get the atomnumbers of atoms within a certain distance of the
        atom with `centeratomnr`.

        Parameters
        ----------
        centeratomnr : int, optional
            Atomnumber from which to search for atoms. One of `centeratomnr` or
            `centerpoint` has to be given.
        centerpoint_frac : int, optional
            Point from which to search for atoms in fractional coordinates.
            One of `centeratomnr` or `centerpoint` has to be given.
        centerpoint_abs : int, optional
            Point from which to search for atoms in absolute coordinates.
            One of `centeratomnr` or `centerpoint` has to be given.
        max_distance : float
            The distance to search within.
        max_distance_xyz : list of float
            Also check for a maximum distance along a certain direction.
            Uses the respective component of the distance vector. If :const:`None`,
            the *max_distance* is used.
        min_distance_xyz : list of float
            Sets the minimum search distance along a certain direction.
            Uses the respective component of the distance vector. If :const:`None`,
            0 Å is used.
        atom_types : list of str
            Atomtypes to search for.

        Returns
        -------
        atomnumber : int
            Atomnumber with a distance lower than the specified *distance*.
        newatom : :class:`morty.atomistic.Atom`
            Translated atom as returned by
            :class:`morty.atomistic.Cell.get_shortest_distance()`.
        connection vector : np.ndarray
            The vector connecting atom1 and the translated atom2.

        Notes
        -----
        This function can only take every atom into account once, meaning you
        will not get more than one translated images, but only the nearest one
        (which CAN be a translated image)

        """
        if centeratomnr is not None:
            myat = self.get_atom(centeratomnr)
        elif centerpoint_frac is not None:
            myat = Atom(self.lattice_cart, position_frac=centerpoint_frac,
                        atom_type='H')
        elif centerpoint_abs is not None:
            myat = Atom(self.lattice_cart, position_abs=centerpoint_abs,
                        atom_type='H')
        else:
            raise RuntimeError('No center given.')
        if atom_types is None:
            if max_distance_xyz is None:
                return [(atnr + 1, self.get_shortest_distance(
                    myat, self.atoms[atnr], returnnewatom2=False)[3])
                        for atnr in range(0, len(self.atoms), 1)
                        if (max_distance > self.get_shortest_distance(
                            myat, self.atoms[atnr], returnnewatom2=False)[0] >
                            min_distance)]
            max_distance_xyz = [max_distance if x is None else x for x in
                                max_distance_xyz]
            return [(atnr + 1, self.get_shortest_distance(
                myat, self.atoms[atnr], returnnewatom2=False)[3])
                    for atnr in range(0, len(self.atoms), 1)
                    if (max_distance > self.get_shortest_distance(
                        myat, self.atoms[atnr], returnnewatom2=False)[0] >
                        min_distance and
                        (np.array(min_distance_xyz) <= np.abs(
                            self.get_shortest_distance(
                                myat, self.atoms[atnr],
                                returnnewatom2=False)[2]))
                        .all() is True and
                        (np.abs(self.get_shortest_distance(
                            myat, self.atoms[atnr],
                            returnnewatom2=False)[2]) <
                         np.array(max_distance_xyz)).all() is True)]
        else:
            if max_distance_xyz is None:
                return [(atnr + 1,
                         self.get_shortest_distance(myat, self.atoms[atnr],
                                                    returnnewatom2=False)[3])
                        for atnr in range(0, len(self.atoms), 1)
                        if (self.get_shortest_distance(
                            myat, self.atoms[atnr], returnnewatom2=False)[0] <
                            max_distance and
                            self.atoms[atnr].atom_type in atom_types)]
            max_distance_xyz = [max_distance if x is None else x for x in
                                max_distance_xyz]
            return [(atnr + 1, self.get_shortest_distance(
                myat, self.atoms[atnr], returnnewatom2=False)[3])
                    for atnr in range(0, len(self.atoms), 1)
                    if (min_distance < self.get_shortest_distance(
                        myat, self.atoms[atnr], returnnewatom2=False)[0] <
                        max_distance and self.atoms[atnr].atom_type in
                        atom_types and
                        (np.array(min_distance_xyz) <= np.abs(
                            self.get_shortest_distance(
                                myat, self.atoms[atnr],
                                returnnewatom2=False)[2]))
                        .all() is True and
                        (np.abs(self.get_shortest_distance(
                            myat, self.atoms[atnr],
                            returnnewatom2=False)[2]) <
                         np.array(max_distance_xyz)).all() is True)]

    def get_atoms(self, names=None, relative_numbering=False):
        """
        Returns a range of atoms.

        Returns the atoms identified by `names` or all atoms.

        Parameters
        ----------
        names: list of (string or int)
            Usually the absolute numbering of the atom in the order as in the
            input file. If relative_numbering is :const:`True`, the relative numbering
            is used. E.g. 'C4' is the fourth carbon atom in the order as in
            the input file and not the fourth atom.
        relative_numbering : bool
            If to use relative numbering. See `name`.

        Returns
        -------
        atom : list of :class:`morty.atomistic.Atom`
            The atom instance requested.

        """
        if names is None:
            return self.atoms
        else:
            if relative_numbering is True:
                myindices = []
                for myname in names:
                    myatomstocheck = [a for a in self.atoms if a.atom_type ==
                                      ''.join(re.findall("[a-zA-Z]+", myname))]
                    myindices.append(self.atoms.index(
                        myatomstocheck[int(''.join(filter(
                            lambda x: x.isdigit(), str(myname)))) - 1]))
                return [self.atoms[i] for i in myindices]

            return [self.atoms[int(''.join(filter(
                lambda x: x.isdigit(), str(myname)))) - 1]
                    for myname in names]

    def get_cell_atoms_xyz(self, header=False):
        """
        Returns a string holding the cartesian coordinates in Å of all
        atoms one per line.

        Parameters
        ----------
        header : bool
            If :const:`True`, a fully fledged *xyz* file including the header is
            returned. If :const:`False`, only the bare cartesian coordinates are
            returned.

        Returns
        -------
        xyz_string : string
            The string containing all cartesian coordinates or the full xyz
            file.

        """
        cell_format = str()
        if header is True:
            cell_format += str(len(self.atoms)) + '\n'
            cell_format += 'Created using morty \n'
        for myatom in self.atoms:
            cell_format += (' ' + myatom.atom_type + ' ' +
                            str(myatom.position_abs[0] * 1e10) +
                            ' ' + str(myatom.position_abs[1] * 1e10) +
                            ' ' + str(myatom.position_abs[2] * 1e10) +
                            '\n')
        return cell_format

    def get_cell_format(self):
        """
        Returns the cartesian cell parameters and absolute atomic coordinates
        in cell format.

        Returns
        -------
        cell_format: str
            The content of the cellfile corresponding to the `Cell` instance.

        """
        cell_format = '%block LATTICE_CART\n' + 'ang\n'
        cell_format += (str(self.lattice_cart[0, 0] * 1e10) + ' ' +
                        str(self.lattice_cart[0, 1] * 1e10) + ' ' +
                        str(self.lattice_cart[0, 2] * 1e10) + '\n')
        cell_format += (str(self.lattice_cart[1, 0] * 1e10) + ' ' +
                        str(self.lattice_cart[1, 1] * 1e10) + ' ' +
                        str(self.lattice_cart[1, 2] * 1e10) + '\n')
        cell_format += (str(self.lattice_cart[2, 0] * 1e10) + ' ' +
                        str(self.lattice_cart[2, 1] * 1e10) + ' ' +
                        str(self.lattice_cart[2, 2] * 1e10) + '\n')
        cell_format += '%endblock LATTICE_CART\n\n' + '%block POSITIONS_FRAC\n'
        for myatom in self.atoms:
            cell_format += (' ' + myatom.atom_type + ' ' +
                            str(myatom.position_frac[0]) +
                            ' ' + str(myatom.position_frac[1]) + ' ' +
                            str(myatom.position_frac[2]) + '\n')
        cell_format += '%endblock POSITIONS_FRAC\n'
        return cell_format

    def get_cell_format_abc_abs(self):
        """
        Returns the conventional cell parameters and absolute atomic
        coordinates in cell format.

        Returns
        -------
        cell_format: str
            The content of the cellfile corresponding to the `Cell` instance.

        """
        cell_format = '%block LATTICE_ABC\n' + 'ang\n'
        cell_format += (str(self.lattice_abc[0] * 1e10) + ' ' +
                        str(self.lattice_abc[1] * 1e10) + ' ' +
                        str(self.lattice_abc[2] * 1e10) + '\n' +
                        str(np.degrees(self.lattice_abc[3])) + ' ' +
                        str(np.degrees(self.lattice_abc[4])) + ' ' +
                        str(np.degrees(self.lattice_abc[5])) + ' ' +
                        '\n')
        cell_format += '%endblock LATTICE_ABC\n\n' + '%block POSITIONS_ABS\n'
        for myatom in self.atoms:
            cell_format += (' ' + myatom.atom_type + ' ' +
                            str(myatom.position_abs[0] * 1e10) +
                            ' ' + str(myatom.position_abs[1] * 1e10) +
                            ' ' + str(myatom.position_abs[2] * 1e10) +
                            '\n')
        cell_format += '%endblock POSITIONS_ABS\n'
        return cell_format

    @staticmethod
    def get_dihedral(atom1, atom2, atom3, atom4):
        """
        Calculate the dihedral angle spanned by four atoms.
        `atom2` and `atom3` are the ones in the middle.

        Parameters
        ----------
        atom1, atom2, atom3, atom4 : :class:`morty.atomistic.Atom`
            Atoms, between which the angle will be calculated.

        Returns
        -------
        dihedral : float
            The dihedral spanned, in rad.

        Notes
        -----
        The viewing direction has been changed w.r.t. Wikipedia, to adopt the
        definition Jmol uses when calculating dihedrals (using the far end
        vector to define the rotational direction)

        """
        vec1 = atom1.position_abs - atom2.position_abs
        vec2 = atom3.position_abs - atom2.position_abs
        vec3 = atom3.position_abs - atom4.position_abs
        return math.atan2((norm(vec2) * vec1) @ np.cross(vec2, vec3),
                          np.cross(vec1, vec2) @ np.cross(vec2, vec3))

    @staticmethod
    def get_distance(atom1, atom2):
        """
        Calculate the distance between two atoms.

        Parameters
        ----------
        atom1, atom2 : :class:`morty.atomistic.Atom`
            Atoms, between which the distance will be calculated.

        Returns
        -------
        distance : float
            The distance in m.

        """
        return np.linalg.norm(atom1.position_abs - atom2.position_abs)

    def get_distances_upto(self, atomset1, atomset2, upto):
        """
        Calculates the distances between two atom sets.

        Calculates the distances between two given sets of atoms up to a
        certain length. All possible connections between both atom sets are
        considered.

        Parameters
        ----------
        atomset1 : array of :class:`morty.atomistic.Atom`
            List of atoms, for which the distance to `atomset2` should be
            evaluated.
        atomset2 : array
            See `atomset2`.
        upto : float
            Maximum length in Angstrom upto which the distances should be
            evaluated.

        Returns
        -------
        distances : list of [(atom1, (atom2, translatedPosition), distance)]
            Array of all possible distances with information about the atoms.
            *atom1* and *atom2* are :class:`morty.atomistic.Atom` instances,
            *translatedPosition* is the (possibly translated) vector of the
            *atom2* (*atom1* is always kept fixed) and *distance* is the
            distance in meters.

        """
        distances_array = []

        # put all required atoms inside unit cell
        for myatom in np.concatenate([np.array(atomset1), np.array(atomset2)]):
            new_position = myatom.position_frac
            for i in range(len(new_position)):
                while new_position[i] >= 1 or new_position[i] < 0:
                    new_position[i] -= np.floor(new_position[i])
            myatom.position_frac = new_position

        for atom1 in atomset1:
            coord_atom1 = atom1.position_frac
            for atom2 in atomset2:
                translations = []
                coord_atom2 = atom2.position_frac
                mymax = np.ceil(upto / np.array(self.lattice_abc)[0:3])
                for i in range(int((mymax[0] * 2 + 1) * (mymax[1] * 2 + 1) *
                                   (mymax[2] * 2 + 1))):
                    translations.append([i % (mymax[0] * 2 + 1) - mymax[0],
                                         math.floor(i / (mymax[0] * 2 + 1)) %
                                         (mymax[1] * 2 + 1) - mymax[1],
                                         math.floor(i / ((mymax[0] * 2 + 1) *
                                                         (mymax[1] * 2 + 1)) %
                                                    (mymax[2] * 2 + 1)) - mymax[2]])

                for translate in translations:
                    # The order atom2 - atom1 is important! Other-
                    # wise we will invert the translation direction
                    distance = np.linalg.norm(
                        (coord_atom2 - coord_atom1 + translate) @ self.lattice_cart)
                    if distance <= upto and distance != 0:
                        distances_array.append((
                            atom1,
                            # This atom2 is not translated.
                            (atom2,
                             # Output the translated absolute
                             # coordinates as an 1-d array.
                             np.array((coord_atom2 + translate) @
                                      self.lattice_cart)),
                            distance))
        return distances_array

    def get_relative_atom_number(self, myatom):
        """
        Get the relative atom number of an atom instance.

        Returns the relative atom number of the given atom instance within the
        element.

        Parameters
        ----------
        atom : :class:`morty.atomistic.Atom` or int
            An instance of atom, or the absolute atom number.

        Returns
        -------
        relative_index : int
            The relative index of the atom.

        """
        if isinstance(myatom, Atom):
            return [a for a in self.atoms
                    if a.atom_type == myatom.atom_type].index(myatom) + 1

        if isinstance(myatom, int):
            return [a for a in self.atoms
                    if a.atom_type == self.get_atom(myatom).atom_type].\
                index(self.get_atom(myatom)) + 1

        return None

    def get_second_moment(self, atomset1, atomset2, i_atomset2,
                          max_shell=(1, 1, 1), max_dist=None):
        """
        Function to calculate a distance sum and second moments for given sets
        of atoms.

        Parameters
        ----------
        atomset1 : list of :class:`morty.atomistic.Atom`
            The atomset representing the receiving sort of atom.
            E.g. for 1H-13C CP, this will be C. The second moments are
            averaged over this set. All atoms in this set must have the same
            `atom_type`.
        atomset2 : list of :class:`morty.atomistic.Atom`
            The atomset representing the transferring sort of atom.
            For 1H-13C CP, this will be H. All atoms in this set must have the
            same `atom_type`.
        i_atomset1 : float
            Total angular momentum of spins sort in `atomset1`.
        max_shell : list of int
            The number of cells along x,y and z to sum up over.
            Defaults to [1,1,1] and is overridden by `max_dist`.
        max_dist : float
            The maximum distance up to which to sum up. If set, `max_shell`
            is calculated automatically. This ensures that summing up is
            done within a sphere around the central respective atom in
            `atomset1`.

        Returns
        -------
        distance sum : float
            np.sum(alldistances**(-6.0))
        second moment : float
            Well... the second moment
        max dist : float
            The maximum distance up to which summing up took place.
        max shells : list of int
            Number of shells taken into account along x, y and z.

        Notes
        -----
        Supply only atoms with the same `atom_type` for `atomset1` and `atomset2`,
        respectively.

        The second moment is defined as

        .. math::
            M_2^{XY} = \\frac{16}{15} * \\frac{1}{(2\\pi)^2} * I(I+1) *
            (\\gamma_I\\gamma_S\\hbar\\frac{\\mu_0}{4\\pi})^2
            \\sum_{i}({r_i^{XY}})^{-6}

        This means, here the second moment IS divided by :math:`2\\pi` as
        Lena did it in ancient times, and is NOT defined according to Bertmer
        et. al. :cite:`Bertmer2000` but according to van Vleck
        :cite:`VanVleck1948a`.

        Examples
        --------
        Assume you have a cell and want to calculate the 1H->13C second moments
        for a 3x3x3 supercell and aromatic carbons. ::

            mycell = atomistic.Cell('myexample.cell')
            mycell.mod_put_atoms_into_cell()
            mycell.det_bonds()
            mycell.det_groups()
            atomset2 = [atom for atom in mycell if (a.atom_type == 'C' and
                        a.properties['chem_group'] == 'aromatic')]
            atomset2 = [atom for atom in mycell if a.atom_type == 'H']
            m2_aromatic = get_second_moment(mycell, atomset1, atomset2,
                                            max_shell=[3, 3, 3])

        If you calculate the second moments for more than one chemical sort,
        you can now get the ccps to judge the quality of your structure
        proposals :cite:`PCCP_2009_11_3522_Seyfarth`.

        A fully worked out example of this is also served within senkeripynb on
        the example of Melon.

        """
        if max_dist is not None:
            x_max_shell = int(
                np.ceil(max_dist / max(self.lattice_cart[:, 0])))
            y_max_shell = int(
                np.ceil(max_dist / max(self.lattice_cart[:, 1])))
            z_max_shell = int(
                np.ceil(max_dist / max(self.lattice_cart[:, 2])))
        else:
            x_max_shell = max_shell[0]
            y_max_shell = max_shell[1]
            z_max_shell = max_shell[2]
        mytranslations = np.array([[x, y, z]
                                   for z in
                                   range(-z_max_shell, z_max_shell + 1, 1)
                                   for y in
                                   range(-y_max_shell, y_max_shell + 1, 1)
                                   for x in
                                   range(-x_max_shell, x_max_shell + 1, 1)]
                                 ).reshape((2 * x_max_shell + 1,
                                            2 * y_max_shell + 1,
                                            2 * z_max_shell + 1, 3))
        mydistances = np.zeros(
            len(atomset1) * len(atomset2) * mytranslations.shape[0] *
            mytranslations.shape[1] * mytranslations.shape[2]
        ).reshape((len(atomset1), len(atomset2),
                   mytranslations.shape[0], mytranslations.shape[1],
                   mytranslations.shape[2]))
        for i1, atom1 in enumerate(atomset1):
            for i2, atom2 in enumerate(atomset2):
                mydistances[i1][i2] = np.linalg.norm(
                    (atom2.position_frac + mytranslations -
                     atom1.position_frac) @
                    np.asarray(self.lattice_cart), axis=3)
        if max_dist is None:
            return [np.sum(mydistances ** (-6.0)),
                    np.sum(mydistances ** (-6.0)) * 16 / 15 / (2 * np.pi)**2 *
                    i_atomset2 * (i_atomset2 + 1) *
                    (atomset1[0].get_gyromagnetic_ratio() * atomset2[
                        0].get_gyromagnetic_ratio() *
                     scipy.constants.hbar * scipy.constants.mu_0 / 4 /
                     scipy.constants.pi) ** 2 / len(atomset1),
                    np.max(mydistances),
                    [x_max_shell, y_max_shell, z_max_shell]]
        return [np.sum(mydistances[
            np.where(mydistances <= max_dist)] ** (-6.0)),
                np.sum(mydistances[
                    np.where(mydistances <= max_dist)] ** (-6.0)) *
                16 / 15 / (2 * np.pi)**2 * i_atomset2 * (i_atomset2 + 1) *
                (atomset1[0].get_gyromagnetic_ratio() * atomset2[
                    0].get_gyromagnetic_ratio() *
                 scipy.constants.hbar * scipy.constants.mu_0 / 4 /
                 scipy.constants.pi) ** 2 / len(atomset1),
                np.max(mydistances[np.where(mydistances <= max_dist)]),
                [x_max_shell, y_max_shell, z_max_shell]]

    def get_shortest_distance(self, atom1, atom2, returnnewatom2=False):
        """
        Calculates the shortest distance between two atoms.

        Calculates the shortest distance between two atoms in the structure,
        considering that this might not be the distance within the unit cell
        but from the unit cell to an adjacent translated cell.

        Parameters
        ----------
        atom1, atom2 : :class:`morty.atomistic.Atom`
        returnnewatom2 : boolean
            If :const:`True`, a copy of `atom2` is returned, with updated coordinates
            respecting the translation yielding the shortest distance.

        Returns
        -------
        distance : float
            Returns the shortest distance between the two specified atoms.
        atom2 : :class:`morty.atomistic.Atom`
            The second atom; if `returnnewatom2`==:const:`True`, the atom is shifted to the
            respective quadrant.
        distance_vector : array
            The distance vector of the two given atoms.
        new_coords : array
            The coordinates of the mirror of atom2 yielding the shortest
            distance.

        Notes
        -----
        There is a very dirty hack in place to take care of the case of one
        coordinate being ``0`` into account. The problem is, if ``0`` is set
        using absolute coordinates, the fractional ones will not be ``0``,
        but something like ``±1e-17``, which is not zero. Therefore the cutoff
        for translating the atoms "back into" the cell are set to -1e5. This
        may lead to problems.

        """
        # All atoms must be in the cell, for this to work. If the user didn't
        # call put_atoms_into_cell(), we do this here:
        atom1, atom2 = self.get_atom(atom1), self.get_atom(atom2)
        for myatom in (atom1, atom2):
            new_position = myatom.position_frac
            for i in range(len(new_position)):
                while new_position[i] >= 1 or new_position[i] < 0:
                    new_position[i] -= np.floor(new_position[i])
            myatom.position_frac = new_position

        mytranslations = (np.array([
            [x, y, z]
            for z in range(-1, 2, 1)
            for y in range(-1, 2, 1)
            for x in range(-1, 2, 1)]).reshape((3, 3, 3, 3)))
        mydistances = np.linalg.norm(
            (atom1.position_frac - (mytranslations + atom2.position_frac))
            @ np.asarray(self.lattice_cart), axis=3)
        mydistindex = np.where(mydistances == np.min(mydistances))
        if returnnewatom2 is True:
            new_atom = copy.deepcopy(atom2)
            new_atom.set_position_frac(
                atom2.position_frac + mytranslations[mydistindex][0])
            return [mydistances[mydistindex][0], new_atom,
                    new_atom.position_abs - atom1.position_abs,
                    new_atom.position_abs]
        return [mydistances[mydistindex][0], atom2,
                (atom2.position_frac - atom1.position_frac +
                 mytranslations[mydistindex][0])
                @ np.asarray(self.lattice_cart),
                (atom2.position_frac + mytranslations[mydistindex][0]
                ) @ np.asarray(self.lattice_cart)]

    def get_supercell_atom_offset(self, position, supercell_size):
        """
        Returns the offset of an atom created by
        :class:`morty.atomistic.Cell.mod_make_supercell()`.

        Parameters
        ----------
        position : array_like
            x, y and z coordinate of the translated cell.
        supercell_size : array_like
            Size (x, y, z) of the supercell.

        """
        return len(self) * (position[2] + position[1] *
                            supercell_size[2] + position[0] *
                            supercell_size[2] * supercell_size[1])

    @property
    def lattice_abc(self):
        """
        Returns the cells axis lengths and angles.

        Returns the length of the cell axis vectors and the angles between
        them.

        Returns
        -------
        abc : tuple
            (a, b, c, alpha, beta, gamma)

        """
        return ([np.linalg.norm(self.lattice_cart[0]),
                 np.linalg.norm(self.lattice_cart[1]),
                 np.linalg.norm(self.lattice_cart[2]),
                 math.acos(self.lattice_cart[1] @ self.lattice_cart[2] /
                           np.linalg.norm(self.lattice_cart[1]) /
                           np.linalg.norm(self.lattice_cart[2])),
                 math.acos(self.lattice_cart[0] @ self.lattice_cart[2] /
                           np.linalg.norm(self.lattice_cart[0]) /
                           np.linalg.norm(self.lattice_cart[2])),
                 math.acos(self.lattice_cart[0] @ self.lattice_cart[1] /
                           np.linalg.norm(self.lattice_cart[0]) /
                           np.linalg.norm(self.lattice_cart[1]))])

    @lattice_abc.setter
    def lattice_abc(self, abc):
        """
        Sets the lattice parameters.

        Sets the lattice parameters using conventional syntax
        (a/b/c alpha/beta/gamma).

        Parameters
        ----------
        abc : tuple
            Tuple of cell axis lengths and angles in radians. In order:
            (a, b, c, alpha, beta, gamma).

        """
        lattice_a = float(abc[0])
        lattice_b = float(abc[1])
        lattice_c = float(abc[2])
        lattice_alpha = float(abc[3])
        lattice_beta = float(abc[4])
        lattice_gamma = float(abc[5])

        # from CASTEP:
        # a is along x (1st constraint)
        # b is on the xy plane (2nd constraint)
        # c is wherever (3rd constraint - positive sqrt)
        lattice_x = np.array([lattice_a, 0, 0])
        lattice_y = np.array([lattice_b * math.cos(lattice_gamma),
                              lattice_b * math.sin(lattice_gamma), 0])
        lattice_z = np.array(
            [lattice_c * math.cos(lattice_beta),
             lattice_c * (math.cos(lattice_alpha) - math.cos(lattice_beta) *
                          math.cos(lattice_gamma)) / math.sin(lattice_gamma),
             math.sqrt(lattice_c ** 2 - (lattice_c * math.cos(lattice_beta)
                                        ) ** 2 - (lattice_c *
                                                  (math.cos(lattice_alpha) -
                                                   math.cos(lattice_beta) *
                                                   math.cos(lattice_gamma)) /
                                                  math.sin(lattice_gamma)) ** 2
                      )])
        self.lattice_cart = np.array([lattice_x, lattice_y, lattice_z])

    def load_castep_bondorders(self, filename=None, foldername=None):
        """
        Reads in bond orders as reported by CASTEP.

        Parameters
        ----------
        filename : str
            Path to the input file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.

        """
        castepfile = [
            x.split() for x in
            filter(None,
                   open(fileio.determine_filename(self, '.castep',
                                                   filename, foldername), 'r')
                   .read().split(' Bond              Population      Length ' +
                                 '(A)')[1].split('=' * 60)[1].split('\n'))]
        self.properties['castep_bondorders'] = []
        for bondorder in castepfile:
            atom1, atom2 = sorted(
                [self.atoms.index(self.get_atom(bondorder[0] + bondorder[1],
                                                relative_numbering=True)) + 1,
                 self.atoms.index(self.get_atom(bondorder[3] + bondorder[4],
                                                relative_numbering=True)) + 1])
            self.properties['castep_bondorders'].append(
                [int(atom1), int(atom2), float(bondorder[5])])

    def load_castep8_magres(self, filename=None, foldername=None,
                            append_atoms=False):
        """
        Loads a CASTEP 8 magres file.

        This reads atom positions and magres information from the new magres
        file format. Beware, quadrupolar tensors and additional information are
        not read yet.

        Parameters
        ----------
        filename : str
            Path to the input file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        append_atoms : bool, optional
            If true, the atoms defined in the loaded cell file are appended to
            the existing `atoms` list. Only makes sense if the cell constants
            are the same or the atoms coordinates are given in cartesians.

        """
        if append_atoms is False:
            self.atoms = []

        myfile = open(fileio.determine_filename(self, '.magres', filename,
                                                 foldername), 'r')
        line = myfile.readline().strip()
        while line != '':
            if '[atoms]' in line:
                while '[/atoms]' not in line:
                    line = myfile.readline().strip()
                    if line.startswith('lattice'):
                        self.lattice_cart = np.array(line.split()[1:10]
                                                    ).astype(np.float
                                                            ).reshape((3, 3)
                                                                     ) * 1e-10
                    if line.startswith('atom'):
                        abs_vector = np.array(line.split()[4:7]
                                             ).astype(np.float) * 1e-10
                        self.atoms.append(Atom(atom_type=line.split()[1],
                                               position_abs=abs_vector,
                                               basis=self.lattice_cart))
            if '[magres]' in line:
                while '[/magres]' not in line:
                    line = myfile.readline().strip()
                    if line.startswith('ms'):
                        line_split = line.split()
                        # the atom label number has a fixed size of 3 characters and therefore
                        # the whitespace between the type and the number may be skipped. quick and
                        # dirty fix.
                        if len(line_split) == 12:
                            myslice = slice(3, 12)
                            myatom = line_split[1] + line_split[2]
                        else:
                            myslice = slice(2, 11)
                            myatom = line_split[1]
                        csa_tensor = np.array(line_split[myslice]
                                             ).astype(np.float
                                                      ).reshape((3, 3)) * - 1
                        self.get_atom(myatom, relative_numbering=True
                                     ).properties['csatensor'] = CSATensor(csa_tensor)
                    if line.startswith('efg'):
                        line_split = line.split()
                        # see comment on magres
                        if len(line_split) == 12:
                            myslice = slice(3, 12)
                            myatom = line_split[1] + line_split[2]
                        else:
                            myslice = slice(2, 11)
                            myatom = line_split[1]
                        efg_tensor = np.array(line_split[myslice]
                                             ).astype(np.float
                                                      ).reshape((3, 3))
                        self.get_atom(myatom, relative_numbering=True
                                     ).properties['efgtensor'] = EFGTensor(
                                         efg_tensor,
                                         atom=self.get_atom(
                                             myatom,
                                             relative_numbering=True))
            line = myfile.readline().strip()
        myfile.close()

    def load_carfile(self, filename=None, foldername=None, append_atoms=False):
        """
        Loads a Materials Studio CAR file.

        Parameters
        ----------
        filename : str
            Path to the input file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        append_atoms : bool, optional
            If :const:`True`, the atoms defined in the loaded car file are appended to
            the existing `atoms` list. Only makes sense if the cell constants
            are the same or the atoms coordinates are given in cartesians.

        """

        file = open(fileio.determine_filename(self, '.car', filename,
                                               foldername), 'r')
        line = file.readline()

        if append_atoms is False:
            self.atoms = []

        while line != '':
            line = line.strip()

            # found lattice parameters in cartesian coordinates
            if line.lower().startswith('pbc') and len(line.split()) == 8:
                abc = np.array(line.split()[1:7]).astype(float) * 1e-10
                if append_atoms is False:
                    abc[3:] = np.radians(np.array(line.split()[4:7]).astype(float))
                    self.lattice_abc = abc
                    line = file.readline()
                break
            line = file.readline()

        while line != '':
            if 'end' in line.lower():
                line = file.readline()
                if 'end' in line.lower():
                    break
            line = line.strip().split()
            abs_vector = np.array([line[1],
                                   line[2],
                                   line[3]]).astype(np.float32) * 1e-10
            # we use the label here for the atom type, because if the atom
            # label is too long, there is not space between the label and the type
            # other option: count characters
            self.atoms.append(Atom(atom_type=''.join(re.findall("[a-zA-Z]+", line[0])),
                                   position_abs=abs_vector,
                                   basis=self.lattice_cart))
            line = file.readline()

        file.close()

    def load_castep_cellfile(self, filename=None, foldername=None,
                             append_atoms=False):
        """
        Loads a cell file.

        By default, loads information about the unit cell and the atom
        parameters. If nonlinear constraints are present, they are written to
        `constraints` (at the moment torsions).
        Additional info like k-point spacing is written to
        `castep_cellfile_additions`, as well as linear constraints.
        This info is not understood by castep and just appended if you
        write a cell file with
        :class:`morty.atomistic.Cell.save_castep()`.

        Parameters
        ----------
        filename : str
            Path to the input file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        append_atoms : bool, optional
            If :const:`True, the atoms defined in the loaded cell file are
            appended to the existing `atoms` list. Only makes sense if the
            cell constants are the same or the atoms coordinates are given
            in cartesians.

        Notes
        -----
        This function also reads in additional input like k-point spacing or
        like, but it is oblivious to what it is reading and treats everything
        as a string.

        """

        file = open(fileio.determine_filename(self, '.cell', filename,
                                               foldername), 'r')
        line = file.readline()

        if append_atoms is False:
            self.atoms = []

        while line != '':
            line = line.strip()

            # found lattice parameters in cartesian coordinates
            if '%block' in line.lower() and 'lattice_cart' in line.lower():
                line = file.readline()
                line = line.strip()

                # first line defines the unit - we only handle angstrom for now
                # we could add conversion functions here, for now we skip the
                # line
                if line.lower().startswith('ang'):
                    line = file.readline()
                    line = line.strip()

                # iterate, first line is x, second y, third z
                xyz = 0
                while '%endblock' not in line.lower():
                    if xyz == 0:
                        lattice_x = np.array(line.split()).astype(float
                                                                 ) * 1e-10
                        xyz = 1
                    elif xyz == 1:
                        lattice_y = np.array(line.split()).astype(float
                                                                 ) * 1e-10
                        xyz = 2
                    elif xyz == 2 and append_atoms is False:
                        lattice_z = np.array(line.split()).astype(float
                                                                 ) * 1e-10
                        self.lattice_cart = np.array([lattice_x, lattice_y,
                                                      lattice_z])

                    line = file.readline()
                    line = line.strip()

            # found lattice parameters a, b, c, alpha, beta, gamma
            elif '%block' in line.lower() and 'lattice_abc' in line.lower():
                line = file.readline()
                line = line.strip()

                if line.lower() == 'ang':
                    line = file.readline()
                    line = line.strip()

                # iterate, first line is abc, second alpha beta gamma
                for i in range(2):
                    if i == 0:
                        abc = np.array(line.split()).astype(float) * 1e-10
                    if i == 1:
                        abg = np.radians(np.array(line.split()).astype(float))
                    line = file.readline()
                    line = line.strip()
                if append_atoms is False:
                    self.lattice_abc = np.concatenate([abc, abg])

            # found atom positions in fractional coordinates
            elif '%block' in line.lower() and 'positions_frac' in line.lower():
                line = file.readline()
                line = line.strip()

                while '%endblock' not in line.lower():
                    line_contents = line.split()
                    if len(line_contents) == 4:
                        self.atoms.append(
                            Atom(atom_type=line_contents[0],
                                 position_frac=np.array([line_contents[1],
                                                         line_contents[2],
                                                         line_contents[3]]
                                                       ).astype(np.float32),
                                 basis=self.lattice_cart))
                    line = file.readline()
                    line = line.strip()

            # found atom positions in absolute coordinates
            elif '%block' in line.lower() and 'positions_abs' in line.lower():
                line = file.readline()
                line = line.strip()

                if line.lower() == 'ang':
                    line = file.readline()
                    line = line.strip()

                while '%endblock' not in line.lower():
                    line_contents = line.split()
                    if len(line_contents) == 4:
                        abs_vector = np.array([line_contents[1],
                                               line_contents[2],
                                               line_contents[3]]
                                             ).astype(np.float32) * 1e-10
                        self.atoms.append(Atom(atom_type=line_contents[0],
                                               position_abs=abs_vector,
                                               basis=self.lattice_cart))
                    line = file.readline()
                    line = line.strip()
            elif ('%block' in line.lower() and
                  'nonlinear_constraints' in line.lower()):
                line = file.readline()
                line = line.strip()
                self.constraints['torsions'] = []
                while '%endblock' not in line.lower():
                    line_contents = line.split()
                    if 'torsion' in line_contents[0]:
                        line_contents.pop(0)
                        self.constraints['torsions'].append([
                            str(line_contents[i] + line_contents[i + 1])
                            for i in range(0, 20, 5)])
                    line = file.readline()
                    line = line.strip()
            # .. todo:: IMPORTANT! implement all possible contraints!
            elif '%block' in line.lower():
                block_end = False
                while block_end is False:
                    self.castep_cellfile_additions[
                        'additional_lines'].append(line)
                    if '%endblock' in line.lower():
                        self.castep_cellfile_additions[
                            'additional_lines'].append('\n')
                        block_end = True
                    else:
                        line = file.readline()
                        line = line.strip()

            elif line != '':
                self.castep_cellfile_additions[
                    re.split('[:=\ ]+', line)[0].lower()] = re.split(
                        '[:=\ ]+', line, 1)[1]

            line = file.readline()
        file.close()

    def load_castep_charges(self, filename=None, foldername=None,
                            scheme='mulliken'):
        """
        Allows to load charges from a *castep* file.

        You can load Mulliken or Hirshfeld charges from a *castep* file.
        Mulliken charges are output of CASTEP by default, while Hirshfeld
        charges have to be specifically requested.

        Parameters
        ----------
        filename : str
            The filename of the file to read the charges from, usually the
            *castep* file. If :const:`None`, will search for a *castep* file
            in the current folder.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `foldername`.
        scheme : {'mulliken', 'hirshfeld'}
            The scheme, for which the charges have been calculated. Defaults to
            Mulliken, for which CASTEP outputs charges by default.

        """
        # Define how to find the charges in terms of scheme : params.
        # params contain: (str - keyword identifying the line in block to
        # search for, int - number of lines to skip 'after' block is found,
        # int - index within read-in line for read in of charge).
        # This function assumes that charges are listed in the same order as
        # the atoms, and, naturally, you have these atoms already in your cell.
        schemes = {
            'hirshfeld': ('Hirshfeld Analysis', 3, 2),
            'mulliken': ('Atomic Populations (Mulliken)', 3, 7)}
        foundcharges = 0
        with open(fileio.determine_filename(self, '.castep', filename,
                                             foldername), 'r') as chargefile:
            for line in chargefile:
                if schemes[scheme][0] in line:
                    foundcharges += 1
                elif foundcharges == (schemes[scheme][1] + 1 +
                                      len(self.atoms) - 2):
                    break
                elif foundcharges > 0:
                    foundcharges += 1
                    if foundcharges > schemes[scheme][1] + 1:
                        try:
                            self.get_atom(line.split()[0] + line.split()[1],
                                          relative_numbering=True).\
                                properties['charge'][scheme] = float(
                                    line.split()[schemes[scheme][2]])
                        except KeyError:
                            self.get_atom(line.split()[0] + line.split()[1],
                                          relative_numbering=True). \
                                properties['charge'] = {scheme: float(
                                    line.split()[schemes[scheme][2]])}
        if foundcharges > 1:
            print("Proud to report successful search and retrieve of charges!")

    def load_castep_energy(self, filename=None, foldername=None,
                           ignore_finished=False, ignore_warning=False):
        """
        Loads the final energy from a *castep* file.

        The read-out energies are saved to `Cell.properties['totalenergy']`
        and `Cell.properties['totalenergy_dispcorr']`. If no related
        arguments are specified, searches the current folder.

        Parameters
        ----------
        filename: string, optional
              Filename of the CASTEP calculation, usually with file
              extension *castep*.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        ignore_finished : bool, optional
            If True, does not warn about unfinished calculations.
        ignore_warning : bool
            If True, ignores any warning messages in the *castep* file.

        Returns
        -------
        calc_finished : bool
            If the calculation is finished.
        calc_warning : bool
            If a warning was printed in the .castep file. If this is the case,
            your calculation maybe didn't converge in the specified number of
            maximum steps in the geometry optimisation or electronic
            minimisation.

        Notes
        -----
        Energies are reported in J.

        The following warning messages of CASTEP are explicitly stated:

            - scf not finished
            - unit cell too small for semiempirical dispersion correction
            - BFGS did not converge, max. number of steps reached

        """

        mytotalenergy = []
        mytotalenergy_dc = []
        calc_finished = False
        calc_warning = False
        known_warning = []
        return_warning = []
        try:
            energyfile = open(fileio.determine_filename(
                self, '.castep', filename, foldername), 'r').readlines()

            for line in energyfile:
                if 'Final energy' in line:
                    mytotalenergy.append(float(line.split()[-2]) *
                                         scipy.constants.physical_constants[
                                             'electron volt'][0])
                elif 'Dispersion corrected final energy' in line:
                    mytotalenergy_dc.append(float(line.split()[-2]) *
                                            scipy.constants.physical_constants[
                                                'electron volt'][0])
                elif 'Total time' in line:
                    calc_finished = True
                elif ('*** There were at least     1 warnings during this run'
                      in line):
                    calc_warning = True
                elif ('*Warning* max. SCF cycles performed but system ' +
                      'has not reached the groundstate.' in line):
                    msg = 'SCF did not converge!'
                    if msg not in known_warning:
                        known_warning += [msg]
                elif ('WARNING: Your unit cell might be too small to get ' +
                      'accurate results for van der Waals' in line):
                    msg = ('\nUnit cell may be too small for reliable sedc ' +
                           'energies.')
                    if msg not in known_warning:
                        known_warning += [msg]
                elif ('BFGS : WARNING - Geometry optimization failed to ' +
                      'converge after' in line):
                    msg = '\nYour BFGS minimisation did not converge.'
                    if msg not in known_warning:
                        known_warning += [msg]
            if ((ignore_finished is True or calc_finished is True) and
                    ignore_warning is True or calc_warning is False):
                try:
                    self.properties['totalenergy'] = mytotalenergy[-1]
                except IndexError:
                    self.properties['totalenergy'] = None
                try:
                    self.properties['totalenergy_dispcorr'] = \
                        mytotalenergy_dc[-1]
                except IndexError:
                    self.properties['totalenergy_dispcorr'] = None
            if ignore_finished is False and calc_finished is False:
                self.properties['totalenergy'] = None
                self.properties['totalenergy_dispcorr'] = None
                return_warning += [str('Calc ' + str(self.cellname) + ' not '
                                       'finished. Set ignore_finished=True ' +
                                       'to ignore this warning.' +
                                       '\n'.join(known_warning))]
            elif calc_finished is False:
                return_warning += [str('Calc ' + str(self.cellname) + ' did '
                                       'not ' +
                                       'finish. The energy reported may not ' +
                                       'be converged.' +
                                       '\n'.join(known_warning))]
            if ignore_warning is False and calc_warning is True:
                self.properties['totalenergy'] = None
                self.properties['totalenergy_dispcorr'] = None
                return_warning += [str('Calc ' + str(self.cellname) +
                                       ' produced a warning in the .castep ' +
                                       'file. You will not be able to read ' +
                                       'final energy. Set ignore_warning=' +
                                       'True to ignore this warning.' +
                                       '\n'.join(known_warning))]
            elif calc_warning is True:
                return_warning += [str('Calc ' + str(self.cellname) + ' ' +
                                       'produced a warning in the .castep ' +
                                       'file. The energy reported may not be' +
                                       ' converged.' +
                                       '\n'.join(known_warning))]
            return calc_finished, calc_warning, ' '.join(return_warning)

        except FileNotFoundError:
            raise FileNotFoundError('Did not find energy file.')

    def load_castep_magres(self, filename=None, foldername=None,
                           quadrupolar_for=None):
        """
        Extract information from a CASTEP *magres* file.

        Extract the chemical shift tensor from a castep magres file. If the
        calculation also yielded the efg tensor, the latter is also retrieved.
        The data read are stored in `atom.properties['csatensor']` and
        `atom.properties['efgtensor']`, respectively.

        Parameters
        ----------
        filename: string, optional
              Filename of the castep *magres* file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        quadrupolar_for : list of str
            For which nuclei to read in quadrupolar tensors, e.g. ['Li'].
            This is necessary as CASTEP chooses quadrupolar nuclei for each
            element it knows of, e.g. 11C, which may not be what you want.
        quadrupolar_masses : list of strings
            Set the mass of the isotope to use for setting the quadrupolar
            tensor. I.e your standard nucleus for NMR is 15N, but only 14N is
            quadrupolar. Doesn't seem to be implemented yet.

        """
        with open(fileio.determine_filename(self, '.magres', filename,
                                             foldername), 'r') as magresfile:
            if '#$magres-abinitio-v1.0' in magresfile.readline():
                self.load_castep8_magres(filename=filename,
                                         foldername=foldername)
        magresfile = open(fileio.determine_filename(
            self, '.magres', filename, foldername), 'r').read().split(
                "Using the following values of the electric quadrupole moment")
        block_csa = magresfile[0].split("============\nAtom:")
        # first get the csa tensor
        if len(magresfile) == 1 or len(magresfile) == 2:
            for index in range(1, len(block_csa)):
                line = block_csa[index].split("\n")
                myatom = line[0].split()[0] + line[0].split()[1]
                csa_tensor = np.array([[line[6].split()[0],
                                        line[6].split()[1],
                                        line[6].split()[2]],
                                       [line[7].split()[0],
                                        line[7].split()[1],
                                        line[7].split()[2]],
                                       [line[8].split()[0],
                                        line[8].split()[1],
                                        line[8].split()[2]]]
                                     ).astype(float) * - 1
                try:
                    self.get_atom(myatom, relative_numbering=True
                                 ).properties['csatensor'
                                             ] = CSATensor(csa_tensor)
                except:
                    raise ValueError("Couldn't find atom " + str(myatom) +
                                     " for setting the csa!")
        if len(magresfile) == 2:
            block_quadrupolar = magresfile[1].split("============\nAtom:")
            for index in range(1, len(block_quadrupolar)):
                line = block_quadrupolar[index].split("\n")
                myatom = line[0].split()[0] + line[0].split()[1]
                quadrtensor = np.array([[line[6].split()[0],
                                         line[6].split()[1],
                                         line[6].split()[2]],
                                        [line[7].split()[0],
                                         line[7].split()[1],
                                         line[7].split()[2]],
                                        [line[8].split()[0],
                                         line[8].split()[1],
                                         line[8].split()[2]]]).astype(float)
                if (quadrupolar_for is None or
                        ''.join(re.findall("[a-zA-Z]+", myatom)) in
                        quadrupolar_for):
                    try:
                        self.get_atom(myatom, relative_numbering=True).\
                            properties['efgtensor'] = (EFGTensor(
                                quadrtensor, atom=self.get_atom(
                                    myatom, relative_numbering=True)))
                    except:
                        raise ValueError("Couldn't find atom " + str(myatom) +
                                         " for setting the quadrupolar tensor!"
                                        )

    def load_castep_paramfile(self, filename=None, foldername=None):
        """
        Load a *param* file used for CASTEP input.

        The parameters loaded are stored in `Cell.castep_paramfile` as a
        dictionary.

        Parameters
        ----------
        filename : string
            The path to the file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.

        """
        file = open(fileio.determine_filename(self, '.param', filename, foldername), 'r')
        castep_parameters = dict()
        for line in file:
            line = line.strip()
            if line != '' and not line.startswith('#'):
                line_split = [x.strip().lower() for x in
                              line.replace("=", " ").replace(":", " ").split()]
                castep_parameters[line_split[0]] = line_split[1]
        self.castep_paramfile = castep_parameters

    def load_cif(self, filename=None, foldername=None):
        """
        Loads a CIF.

        .. todo::
            Not nearly complete, misses many features!

        Parameters
        ----------
        filename: str
            Name of the file to write to.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.

        """

        myfile = open(fileio.determine_filename(self, '.cif', filename,
                                                 foldername), 'r')

        line = myfile.readline()
        lattice_a, lattice_b, lattice_c = None, None, None
        lattice_alpha, lattice_beta, lattice_gamma = None, None, None
        while line != '':
            # skip_readline: in case we are in a loop, we'll only know when
            # the loop ended when we read the next line. In that case, we can't
            # read a new line, as it would be done at the end of this loop.
            skip_readline = False
            line = line.strip()
            # found lattice parameters in cartesian coordinates
            if '_cell_length_a' in line:
                # First split to get the number, then strip the accuracy.
                lattice_a = float(line.split()[1].split('(')[0]) * 1e-10
            if '_cell_length_b' in line:
                lattice_b = float(line.split()[1].split('(')[0]) * 1e-10
            if '_cell_length_c' in line:
                lattice_c = float(line.split()[1].split('(')[0]) * 1e-10
            if '_cell_angle_alpha' in line:
                lattice_alpha = np.radians(float(line.split()[1].split('(')[0]))
            if '_cell_angle_beta' in line:
                lattice_beta = np.radians(float(line.split()[1].split('(')[0]))
            if '_cell_angle_gamma' in line:
                lattice_gamma = np.radians(float(line.split()[1].split('(')[0]))
            if '_symmetry_space_group_name_hall' in line.lower():
                self.spacegroup_hallsymbol = shlex.split(line
                                                        )[1].replace('_', ' ')
                try:
                    self.spacegroup_hallnum = HALL_SYMBOLS.index(
                        self.spacegroup_hallsymbol) + 1
                except ValueError:
                    warnings.warn('Spacegroup not found!')
            if '_symmetry_space_group_name_H-M' in line:
                hm_symbol = shlex.split(line)[1].replace(' ', '')
                for symbol in range(530):
                    if hm_symbol in HM_SYMBOLS[symbol]:
                        self.spacegroup_hallnum = symbol + 1
                        break
                if symbol == 529 and self.spacegroup_hallnum == 1:
                    warnings.warn('Spacegroup not found!')
                else:
                    self.spacegroup_hallsymbol = (HALL_SYMBOLS[symbol])
            if None not in (lattice_a, lattice_b, lattice_c, lattice_alpha,
                            lattice_beta, lattice_gamma):
                self.lattice_abc = [lattice_a, lattice_b, lattice_c,
                                    lattice_alpha, lattice_beta, lattice_gamma]

            if line.startswith('loop_'):
                skip_readline = True
                line = myfile.readline().strip()
                loop_properties = []
                while line.startswith('_'):
                    loop_properties.append(line)
                    line = myfile.readline().strip()
                if '_atom_site_type_symbol' in loop_properties:
                    atom_type = None
                    atom_position_frac = np.empty(3)
                while (line != '' and not line.startswith('_') and not
                       line.startswith('loop_')):
                    loop_line = shlex.split(line)
                    for loop_it, value in enumerate(loop_line):
                        if loop_properties[loop_it] == '_atom_site_type_symbol':
                            atom_type = value
                        if loop_properties[loop_it] == '_atom_site_fract_x':
                            atom_position_frac[0] = float(value.split('(')[0])
                        if loop_properties[loop_it] == '_atom_site_fract_y':
                            atom_position_frac[1] = float(value.split('(')[0])
                        if loop_properties[loop_it] == '_atom_site_fract_z':
                            atom_position_frac[2] = float(value.split('(')[0])
                    # Do we have fractional coordinates?
                    if any('_atom_site_fract' in prop
                           for prop in loop_properties):
                        self.atoms.append(Atom(
                            basis=self.lattice_cart, atom_type=atom_type,
                            position_frac=copy.copy(atom_position_frac)))
                    line = myfile.readline().strip()
            if not skip_readline:
                line = myfile.readline()
        myfile.close()

    def load_xyzfile(self, filename=None, foldername=None,
                     max_x=50e-10, max_y=50e-10, max_z=50e-10):
        """
        Load a standard xyz-file.

        The molecule will be placed in a  50x50x50 Å unit cell.

        Parameters
        ----------
        filename : str, optional
            Path to the input file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        max_x : float, optional
            Length of the x axis for a fake periodic cell, in which the
            molecule is put.
        max_y : float, optional
        max_z : float, optional

        """

        self.lattice_cart = np.diag((max_x, max_y, max_z))

        with open(fileio.determine_filename(self, '.xyz', filename,
                                             foldername), 'r') as myfile:
            data = "".join(myfile.readlines())
        for line in data.split('\n')[2:]:
            if line != '':
                coordinates = line.split()
                abs_vector = np.array([coordinates[1],
                                       coordinates[2],
                                       coordinates[3]]).\
                    astype(np.float32) * 1e-10
                self.atoms.append(Atom(position_abs=abs_vector,
                                       basis=self.lattice_cart,
                                       atom_type=coordinates[0]))

    def load_gaussian_logfile(self, filename=None, foldername=None,
                              standard_orientation=False):
        """
        Loads a Gaussian output file, including the FINAL energy.

        Parameters
        ----------
        filename : str, optional
            Path to the input file.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.
        standard_orientation : bool
            If True, we read the structure in the standard orientation. Beware
            that e.g. CSA tensors are not given in the standard orientation.

        Notes
        -----
        This function is reads the last structure in the log-file by default.

        .. todo:: implement read the output in for z-mat input. Maybe solved
            by always reading the (last) standard orientation.

        """
        self.lattice_cart = np.diag((100e-10, 100e-10, 100e-10))
        if standard_orientation is True:
            block_name = 'Input orientation:'
        else:
            block_name = 'Standard orientation:'

        with open(fileio.determine_filename(self, '.log', filename, foldername
                                            ), 'r') as my_file:
            lines = my_file.readlines()
            for reverse_linenum in reversed(range(len(lines))):
                # read energy
                if ('\\HF=' in lines[reverse_linenum] or
                        'RMSD=' in lines[reverse_linenum]):
                    self.properties['totalenergy'] = (
                        scipy.constants.physical_constants[
                            'atomic unit of energy'][0] *
                        float(re.search(r'(?<=HF=)(.*)(?=\\RMSD)',
                                        lines[reverse_linenum - 1].rstrip() +
                                        lines[reverse_linenum].strip() +
                                        lines[reverse_linenum + 1].lstrip()
                                       ).group(0)))
                # read coordinates
                if block_name in lines[reverse_linenum]:
                    rel_linenum = 5  # skip table header
                    while '---' not in lines[reverse_linenum + rel_linenum]:
                        coordinates = lines[reverse_linenum +
                                            rel_linenum].split()
                        abs_vector = np.array([coordinates[3], coordinates[4],
                                               coordinates[5]]
                                             ).astype(np.float32) * 1e-10
                        try:
                            atom_type = [
                                x for x, y in list(
                                    constants.NUC_CHARGES.items())
                                if y == int(coordinates[1])][0]
                        except IndexError:
                            raise RuntimeError('Atom with charge ' +
                                               str(coordinates[1]) +
                                               ' not known.')
                        self.atoms.append(Atom(position_abs=abs_vector,
                                               basis=self.lattice_cart,
                                               atom_type=atom_type))
                        rel_linenum += 1
                    break  # exit loop

    def load_gaussian_magres(self, filename=None, foldername=None):
        """
        Extract the tensor from a gaussian log file.

        Parameters
        ----------
        filename: string
              Filename of the gaussian calculation, usually with file
              extension *log*.
        foldername : str, optional
            Defines the path to the files. If the first  character is '+',
            the path is handled as a subfolder of `Cell.foldername`.

        """

        nmrfile = open(fileio.determine_filename(
            self, '.log', filename, foldername), 'r').read().split(
                "SCF GIAO Magnetic shielding tensor (ppm):\n")[1].split("\n")
        if len(nmrfile) > 1:
            natoms = len(self.atoms)
            for index in range(natoms):
                shiftdata = nmrfile[index * 5:(index * 5) + 4]
                csatensor = np.array([[shiftdata[1].split()[1],
                                       shiftdata[1].split()[3],
                                       shiftdata[1].split()[5]],
                                      [shiftdata[2].split()[1],
                                       shiftdata[2].split()[3],
                                       shiftdata[2].split()[5]],
                                      [shiftdata[3].split()[1],
                                       shiftdata[3].split()[3],
                                       shiftdata[3].split()[5]]]
                                    ).astype(float)
                csatensor *= -1
                self.atoms[index].properties['csatensor'
                                            ] = CSATensor(csatensor)
                try:
                    self.atoms[index].properties['csatensor'
                                                ] = CSATensor(csatensor)
                except:
                    raise ValueError("Couldn't find atom " + str(index) +
                                     "for setting the csa!")

    def load_gyromagnetic_ratios(self):
        """
        Reads in the gyromagnetic ratios for each atom. See
        :class:`morty.atomistic.Atom.get_gyromagnetic_ratio()`

        """
        for myatom in self.atoms:
            myatom.get_gyromagnetic_ratio()

    def make_p1(self):
        """
        Applies symmetry operations of space group to turn it into P 1.

        This will apply the symmetry operations for the defined space group to
        the atoms, to turn it into P 1 symmetry, where all atoms are included,
        not only the asymmetric unit.

        Notes
        -----
        This uses `Cell.spacegroup_hallnum` to determine the symmetry
        operations.

        It requires spglib to work.

        """
        if SPGLIB_AVAIL is False:
            raise RuntimeError('Spglib not available!')

        self.mod_put_atoms_into_cell()

        rotation = np.zeros((192, 3, 3), dtype='intc', order='C')
        translation = np.zeros((192, 3), dtype='double', order='C')
        num_sym = spglib.symmetry_from_database(rotation, translation,
                                                self.spacegroup_hallnum)
        num_atoms = len(self)
        for symmetry in range(num_sym):
            for atom_num in range(num_atoms):
                new_pos = (rotation[symmetry] @ self.atoms[atom_num
                                                          ].position_frac +
                           translation[symmetry])
                for i in range(len(new_pos)):
                    while new_pos[i] >= 1 or new_pos[i] < 0:
                        new_pos[i] -= np.floor(new_pos[i])
                is_duplicate = False
                for atom2 in self:
                    if np.allclose(new_pos,
                                   atom2.position_frac,
                                   rtol=0, atol=1e-12):
                        is_duplicate = True
                if not is_duplicate:
                    self.atoms.append(copy.deepcopy(self.atoms[atom_num]))
                    self.atoms[-1].position_frac = new_pos
        self.symmetry_operations_applied = True

    def mod_append_atom(self, myatom):
        """
        Adds a given atom to the cell.

        Adds the provided atom instance to the end of the cells list of atoms.
        The basis of the is set to that of this cell, with absolute coordinates
        keeping fixed.

        Parameters
        ----------
        atom : :class:`morty.atomistic.Atom`

        myfunction : list of [function, params, bool]
            A function determining how the new atom is created.

        """
        self.atoms.append(myatom)
        self.atoms[-1].set_basis(self.lattice_cart, fix_abs=True)

    def mod_append_atom_by_function(self, fun, parameters):
        """
        Adds a given atom to the cell.

        Parameters are unpacked and all the parameters are then
        handed over to the respective function `fun`.
        This function comes in handy if you want to automatically build
        models via a rastering mechanism like
        :class:`morty.atomistic.CellModeller` using custom rules.

        Parameters
        ----------
        fun : callable
            A function determining how the new atom is created. This function
            must return a :class:`morty.atomistic.Atom` instance.
        params1 : list
            A set of parameters for `fun`.

        Notes
        -----
        There are certain rules to obey when defining the function, which will
        become important once you use a `CellModeller` with *mod* functions.
        See the documentation of :class:`morty.atomistic.CellModeller` for instructions.
        A good starting point to see how the function works is given in the
        example. Or have a look at :class:`morty.atomistic.Cell.mod_make_nx()`

        Examples
        --------
        First you will want to define a function to add the atom. Say you
        want to add an Xe at [0, 0, 0]. ::

            from morty.atomistic import Atom
            from morty.atomistic import Cell
            import numpy as np
            def my_func(target, my_atom_type, abs_coord, num):
                return Atom(basis=target.lattice_cart,
                            position_abs=abs_coord[num],
                            atom_type=my_atom_type)

        Now you can use that function on a cell. ::

            mycell = Cell()
            mycell.lattice_abc = [10e-10, 10e-10, 10e-10,
                                  pi / 2, pi / 2, pi / 2]))
            mycell.mod_append_atom_by_function(my_func, ['Xe', [[0, 0, 0]], 0])

        """
        self.mod_append_atom(fun(self, *(parameters)))

    def mod_delete_atoms(self, nums=None, atom_type=None):
        """
        Deletes the atoms from the cell instance.

        Allows the user to delete intelligently delete atoms from the cell
        instance either by number or `atom_type`.

        Parameters
        ----------
        nums: array of int
            The numbers of the atoms to delete. Starts counting at 1.
        mytype: str
            If nums is not given, mytype will specify to delete all atoms
            of the specified type.

        Notes
        -----
        The numbers start from 1. Deleting starts from the end of the array.
        Please mind that the number and numbering of the remaining atoms is
        changed upon deletion.

        """
        if nums is not None:
            for i in sorted(nums, reverse=True):
                self.atoms.pop(i - 1)
        elif atom_type is not None:
            for myatom in self.atoms[::-1]:
                if myatom.atom_type == atom_type:
                    self.atoms.remove(myatom)

    def mod_make_supercell(self, supercell=None):
        """
        Creates a supercell.

        Parameters
        ----------
        supercell : list
            Gives the number of cells to expand in ``[x, y, z]``.

        """
        extensions = []
        for extend_x in range(supercell[0]):
            for extend_y in range(supercell[1]):
                for extend_z in range(supercell[2]):
                    extensions.append([extend_x, extend_y, extend_z])
        extensions.remove([0, 0, 0])
        atoms_to_append = []
        for extend in extensions:
            tmp_atoms = copy.deepcopy(self.atoms)
            for atom in tmp_atoms:
                atom.position_frac = atom.position_frac + np.array(extend)
            atoms_to_append.extend(tmp_atoms)
        self.atoms.extend(atoms_to_append)
        self.update_lattice_cart(self.lattice_cart * np.array(supercell
                                                             )[np.newaxis].T,
                                 fix_abs=True)

    def mod_move_atoms(self, whichatoms=None, vector=np.array([1, 0, 0]),
                       distance=1e-10):
        """
        Moves atoms along a given vector.

        Parameters
        ----------
        atoms : array_like
            The numbers of the atoms to be dislocated. If :const:`None`, all
            atoms are moved.
        vector : array_like
            The vector along which the movement takes place.
        distance : float
            The magnitude of the vector to move about.

        """
        # convert to float, since divide doesn't work with int
        vector = np.array(vector).astype(np.float64)
        vector /= np.linalg.norm(vector)
        if not whichatoms:
            whichatoms = range(1, len(self.atoms) + 1)
        for i in whichatoms:
            self.get_atom(i).position_abs = (
                self.get_atom(i).position_abs + distance * vector)

    def mod_move_atoms_frac(self, whichatoms=None,
                            fracvector=np.array([1, 0, 0]), multiplier=1):
        """
        Moves atoms along a given vector in fractional coordinates.

        Parameters
        ----------
        atoms : array of int
            The numbers of the atoms to be dislocated. If :const:`None`, all
            atoms are moved.
        vector : array of floats
            The vector along which the movement takes place.
        multiplier : float
            Defines the factor to multiply the fractional movement vector with.

        """
        if not whichatoms:
            whichatoms = range(1, len(self.atoms) + 1)
        for i in whichatoms:
            self.get_atom(i).position_frac = (self.get_atom(i).position_frac +
                                              np.array(fracvector) *
                                              multiplier)

    def mod_point_reflection(self, atoms, reflection_point):
        """
        Reflect atoms at a given point.

        Parameters
        ----------
        atoms : array_like
            The numbers of the atoms to be dislocated. If :const:`None`, all
            atoms are moved.
        relection_point : array_like
            Point (in cartesian coordinates) at which the atoms will be
            reflected.

        """
        if not atoms:
            atoms = range(1, len(self.atoms) + 1)
        for i in atoms:
            self.get_atom(i).position_abs = (
                (self.get_atom(i).position_abs - reflection_point) @ (
                    np.diag([-1, -1, -1]))) + reflection_point

    def mod_put_atoms_into_cell(self):
        """
        Puts all atoms outside the cell back into the cell.

        Checks if any atoms coordinates are outside the cell parameters and
        puts them back into the cell. This might be usefull for some
        manipulations and should be done before any transformations (mostly
        prepended by *mod* for functions of `Cell`.

        """
        for myatom in self.atoms:
            new_position = myatom.position_frac
            for i in range(len(new_position)):
                while new_position[i] >= 1 or new_position[i] < 0:
                    new_position[i] -= np.floor(new_position[i])
            myatom.position_frac = new_position

    def mod_rotate_with_dynamic_axis(self, whichatoms, atom1, atom2, angle):
        """
        Rotates a given set of atoms about the connection between two atoms.

        Parameters
        ----------
        atoms : array_like
            The numbers of the atoms to be rotated.
        atom1 : int
            The first atom defining the rotation axis.
        atom2 : int
            The second atom defining the rotation axis.
        angle : float (radians)
            The magnitude of the angle to rotate about.

        """
        axis_vector = (self.get_atom(atom2).position_abs -
                       self.get_atom(atom1).position_abs)
        axis_point = self.get_atom(atom1).position_abs

        self.mod_rotate_with_fixed_axis(whichatoms, axis_point, axis_vector,
                                        angle)

    def mod_rotate_with_fixed_axis(self, whichatoms, axis_point, axis_vector,
                                   angle):
        """
        Rotates a given set of atoms around a static axis. The axis is defined
        by a vector and its foot.

        Parameters
        ----------
        whichatoms : array_like
            The numbers of the atoms to be rotated.
        axis_point : array_like
            The foot defining the axis in conjunction with the vector, in
            cartesian coordinates.
        axis_vector : array_like
            The vector defining the orientation of the axis.
        angle : float
            The magnitude of the angle to rotate about in rad.

        """
        for j in whichatoms:
            myatom = self.get_atom(j)
            rebased_vector = myatom.position_abs - axis_point
            axis_vector /= np.linalg.norm(axis_vector)
            rotation_matrix = axis_rotation_matrix(axis_vector, angle)
            rotated_offset = np.asarray((rotation_matrix @
                                         rebased_vector.reshape(3, 1)
                                        ).reshape(1, 3))[0]
            myatom.set_position_abs(rotated_offset + axis_point)

            if 'csatensor' in myatom.properties:
                myatom.properties['csatensor'] = CSATensor(
                    rotation_matrix @ myatom.properties['csatensor'].tensor @
                    rotation_matrix.I)
            if 'efgtensor' in myatom.properties:
                myatom.properties['efgtensor'] = EFGTensor(
                    rotation_matrix @ myatom.properties['efgtensor'].tensor @
                    rotation_matrix.I, atom=myatom)

    def save_castep(self, filename=None, write_constraints=True, abs_coordinates=False):
        """
        Saves a CASTEP *cell* file.

        A *cell* file is written, which contains unit cell parameters,
        atomic coordinates as well as contraints or additional lines in the
        cell file as defined within the cell instance.

        .. todo::

            * include other constraints but plain fixation of position

        Parameters
        ----------
        filename : str
            Path to the output file.
        write_constraints : bool
            Whether to write constraints defined in self.constraints to the
            cellfile.
        abs_coordinates : bool
            Whether to save atom positions in fractional or absolute
            coordinates.

        Notes
        -----
        Beware, if the cell has symmetry, this transforms it to P1.

        """
        if not filename:
            if self.cellname:
                filename = self.foldername + '/' + self.cellname + '.cell'
            else:
                filename = self.foldername + '/' + 'myfile.cell'
        with open(filename, 'w') as myfile:
            num_sym = 0
            if self.spacegroup_hallnum != 1 and self.symmetry_operations_applied is not True:
                rotation = np.zeros((192, 3, 3), dtype='intc', order='C')
                translation = np.zeros((192, 3), dtype='double', order='C')
                num_sym = spglib.symmetry_from_database(rotation, translation,
                                                        self.spacegroup_hallnum)
                # we don't write it at the top, because jmol can't read it.
                # we need to make it P1 here though, because otherwise not all
                # atoms will be written
                self.make_p1()
            if abs_coordinates is True:
                myfile.write(self.get_cell_format_abc_abs())
            else:
                myfile.write(self.get_cell_format())
            if write_constraints is True:
                nonlinconstr = []
                for k in self.constraints:
                    if k == 'torsions':
                        for constraint in self.constraints[k]:
                            nonlinconstr.append(
                                'torsion ' + ' '.join(
                                    [str(re.sub('\d', "", c) + " " +
                                         re.sub("\D", "", c) + " 0 0 0 "
                                         ) for c in constraint]))
                if nonlinconstr != []:
                    myfile.write('\n%block NONLINEAR_CONSTRAINTS\n' +
                                 '\n'.join(nonlinconstr) +
                                 '\n%endblock NONLINEAR_CONSTRAINTS\n')

                constr = ''
                constr_counter = 0
                if 'fix_position' in self.constraints.keys():
                    for atom in self.constraints['fix_position'][
                            'atomnumbers']:
                        for i in range(3):
                            myconstrain = ['0.0', '0.0', '0.0']
                            myconstrain[i] = '1.0'
                            constr_counter += 1
                            constr += (
                                str(constr_counter) + '   ' +
                                self.get_atom(atom).atom_type + '   ' +
                                str(self.get_relative_atom_number(int(atom))) +
                                '   ' + ' '.join([str(x) for x in myconstrain]) +
                                '\n')
                if 'fix_direction' in self.constraints.keys():
                    for atom in self.constraints['fix_direction'][
                            'atomnumbers']:
                        for i, vector in enumerate(self.constraints[
                                'fix_direction']['vectors']):
                            constr_counter += 1
                            constr += (
                                str(constr_counter) + "   " +
                                self.get_atom(atom).atom_type + "   " +
                                str(self.get_relative_atom_number(atom)) +
                                '   ' + ' '.join([
                                    str(x) for x in vector]) +
                                '\n')
                if constr != '':
                    myfile.write('\n%block IONIC_CONSTRAINTS\n' + constr +
                                 '%endblock IONIC_CONSTRAINTS\n')

            if num_sym != 0:
                myfile.write('%BLOCK SYMMETRY_OPS\n')
                for sym in range(num_sym):
                    myfile.write(str(rotation[sym]).replace(
                        '[', '').replace(']', '') + '\n')
                    myfile.write(str(translation[sym]).replace(
                        '[', '').replace(']', '') + '\n')
                myfile.write('%ENDBLOCK SYMMETRY_OPS\n')
            for key in self.castep_cellfile_additions:
                if key != 'additional_lines':
                    myfile.write(key + ' : ' +
                                 self.castep_cellfile_additions[key] + '\n')
                else:
                    myfile.write("\n" + "\n".join(
                        self.castep_cellfile_additions['additional_lines']))

    def save_cif(self, filename=None):
        """
        Saves a CIF.

        Parameters
        ----------
        filename : str
            Path to the output file.

        """
        if not filename:
            if self.cellname:
                filename = self.foldername + '/' + self.cellname + '.cif'
            else:
                filename = self.foldername + '/' + 'myfile.cif'
        with open(filename, 'w') as myfile:
            # Materials Studio needs a site label and cell parameters are only
            # allowed to have a certain number of decimal numbers
            myfile.write('data_\n')
            myfile.write('_cell_length_a ' +
                         '{:.4f}'.format(self.lattice_abc[0] * 1e10) + '\n')
            myfile.write('_cell_length_b ' +
                         '{:.4f}'.format(self.lattice_abc[1] * 1e10) + '\n')
            myfile.write('_cell_length_c ' +
                         '{:.4f}'.format(self.lattice_abc[2] * 1e10) + '\n')
            myfile.write('_cell_angle_alpha ' +
                         '{:.4f}'.format(math.degrees(self.lattice_abc[3])) +
                         '\n')
            myfile.write('_cell_angle_beta ' +
                         '{:.4f}'.format(math.degrees(self.lattice_abc[4])) +
                         '\n')
            myfile.write('_cell_angle_gamma ' +
                         '{:.4f}'.format(math.degrees(self.lattice_abc[5])) +
                         '\n')
            myfile.write('_symmetry_space_group_name_hall ' +
                         str(self.spacegroup_hallsymbol.replace(' ', '_')) +
                         '\n')
            myfile.write('_symmetry_space_group_name_H-M ' +
                         HM_SYMBOLS[self.spacegroup_hallnum - 1].split(';')[0] + '\n')
            myfile.write('loop_\n_atom_site_label\n_atom_site_type_symbol\n' +
                         '_atom_site_fract_x\n_atom_site_fract_y\n' +
                         '_atom_site_fract_z\n')
            relative_numbers = {}
            for myatom in self:
                try:
                    relative_numbers[myatom.atom_type] += 1
                    relative_number = relative_numbers[myatom.atom_type]
                except KeyError:
                    relative_number = 1
                    relative_numbers[myatom.atom_type] = 1
                myfile.write(myatom.atom_type + str(relative_number) + ' ' +
                             myatom.atom_type + ' ' +
                             str(myatom.position_frac[0]) + ' ' +
                             str(myatom.position_frac[1]) + ' ' +
                             str(myatom.position_frac[2]) + '\n')


    def save_gaussian(self, filename=None):
        """
        Saves a GAUSSIAN *com* file.

        The necessary linebreaks and blank lines
        are inserted automatically. Can also handle Multi-step jobs. The
        latter is for example necessary for calculating NMR for a preceding
        geometry optimisation, using the same wavefunctions as came out from
        the last step of the geometry optimisation instead of just using the
        same geometry.

        Parameters
        ----------
        filename : string
            Filename for the output file.

        Notes
        -----
        The parameters used for the export of the *com* file are taken from
        `Cell.gaussian_params`.

        """
        if not filename:
            if self.cellname:
                filename = self.foldername + '/' + self.cellname + '.com'
            else:
                filename = self.foldername + '/' + 'myfile.com'

        params = self.gaussian_params
        if not params:
            params = {'filename': "myinput.com", 'method': "pbe1pbe",
                      'basis_set': "6-31G", 'jobtype': 'opt',
                      'remaining_route_section': "scf=tight",
                      'charge_and_multiplicity': "0 1",
                      'title': "fancy_default_title", 'link0_section': None,
                      'multi_step_route_section': None,
                      'multi_step_titles': None, 'write_checkfile': 0}
        with open(filename, 'w') as myfile:
            if params['link0_section'] is not None:
                myfile.write(params['link0_section'] + "\n")
            if (params['write_checkfile'] == 1 or
                    params['write_checkfile'] == 3 or
                    params.get('multi_step_route_section') is not None):
                myfile.write("%Chk=" + filename.split(".")[0].split("/")[-1] +
                             ".chk\n")
            myfile.write("#" + params['method'])
            if params['basis_set'] is not None or params['basis_set'] != '':
                myfile.write("/" + params['basis_set'] + " ")
            myfile.write(params['jobtype'] + " ")
            if params['remaining_route_section'] is not None:
                myfile.write(params['remaining_route_section'])
            if (params['write_checkfile'] == 2 or
                    params['write_checkfile'] == 3):
                myfile.write(" " + "formcheck")
            myfile.write("\n\n")
            myfile.write(params['title'] + "\n\n")
            myfile.write(params['charge_and_multiplicity'] + "\n")
            myfile.write(self.get_cell_atoms_xyz())
            myfile.write("\n")
            if params.get('multi_step_route_section') is not None:
                for i, multi in enumerate(params['multi_step_route_section']):
                    myfile.write('--Link1--\n')
                    myfile.write('%Chk=' +
                                 filename.split(".")[0].split("/")[-1] +
                                 '.chk\n')
                    myfile.write('%NoSave\n')
                    myfile.write(params['method'] + '/' + params['basis_set'] +
                                 ' ' + multi + '\n\n')
                    if params['multi_step_titles'] is not None:
                        myfile.write(params['multi_step_titles'][i] + '\n\n')
                    else:
                        myfile.write(params['title'] + '\n\n')

    def save_paramfile(self, filename=None):
        """
        A CASTEP *param* file is written.

        Writes out a CASTEP *param* file, containing the information about the
        job to perform as contained in the `Cell` instance.

        Parameters
        ----------
        filename: str
            The filename to write to. Defaults to 'cellname.param', where
            `cellname` is a property of `Cell`, if the latter is not defined,
            to 'myfile.param'.

        """
        if not filename:
            if self.cellname:
                filename = self.foldername + '/' + self.cellname + '.param'
            else:
                filename = self.foldername + '/' + 'myfile.param'
        with open(filename, 'w') as file:
            for key in self.castep_paramfile:
                if key == 'charge' and 'charge' in self.properties:
                    file.write(str(key) + " = " +
                               str(self.properties['charge']) + "\n")
                else:
                    file.write(str(key) + " = " +
                               str(self.castep_paramfile[key]) + "\n")

    def save_mol(self, filename, supercell='1 1 1'):
        """
        Saves a MOL file as required by CADSS.

        Warning: this might not be the standard way. This is used for input
        files for the CADSS forcefield program.

        """
        with open(filename, 'w') as file:
            file.write('# Number of unit cells :\n  ' + supercell +
                       '\n# MOM_Fundcell_Info: Listed\n')
            file.write(str(self.lattice_abc[0] * 1e10) + ' ' +
                       str(self.lattice_abc[1] * 1e10) + ' ' +
                       str(self.lattice_abc[2] * 1e10) + '\n' +
                       str(math.degrees(self.lattice_abc[3])) + ' ' +
                       str(math.degrees(self.lattice_abc[4])) + ' ' +
                       str(math.degrees(self.lattice_abc[5])) + '\n' +
                       '0.0 0.0 0.0\n')
            file.write('# Lower and Upper Bond Tolerance For Atoms Connectivity\n 0.6 1.164\n' +
                       '# Coord_Info: Listed Cartesian Rigid\n')
            file.write(str(len(self)) + '\n')

            for i, myatom in enumerate(self.atoms):
                file.write('  ' + str(i + 1) + ' ' + str(myatom.position_abs[0] * 1e10) +
                           ' ' + str(myatom.position_abs[1] * 1e10) + ' ' +
                           str(myatom.position_abs[2] * 1e10) + ' ' +
                           str(myatom.properties.get('chem_group')) + ' ' + myatom.atom_type + '\n')

    def save_xyz(self, filename=None):
        """
        Saves the structure as a standard xyz file, naturally omitting the
        unit cell.

        Parameters
        ----------
        filename : string
            Path to the file.

        """
        if not filename:
            if self.cellname:
                filename = self.foldername + '/' + self.cellname + '.xyz'
            else:
                filename = self.foldername + '/' + 'myfile.xyz'
        with open(filename, 'w') as file:
            file.write(self.get_cell_atoms_xyz(header=True))

    def save_zmatrix(self, atom_order, style='TOPAS', filename=None):
        """
        Saves (part of) the structure as a z-matrix.

        Saves the structure as a z-matrix, based on a given order of atoms. For
        now, it renames atoms to start each type at 1 and always uses the three
        previous atoms as reference.

        Parameters
        ----------
        atom_order : array_like
            Atoms which to include in the z-matrix. Order is important here.
            Absolute numbering.
        style : {'TOPAS', ''}
            If 'TOPAS', output will be in TOPAS format.
        filename : string, optional
            If given, saves the output to 'filename'. Otherwise the z-matrix is
            returned as a string.

        """
        atomlist = [self.get_atom(i) for i in atom_order]
        output = ''
        atom_count = {}
        atom_names = [None] * len(atomlist)
        if style == 'TOPAS':
            output += 'rigid\n'
        for i, atom in enumerate(atomlist):
            try:
                atom_count[atom.atom_type] += 1
            except KeyError:
                atom_count[atom.atom_type] = 1
            atom_names[i] = atom.atom_type + str(atom_count[atom.atom_type])

            if style == 'TOPAS':
                output += 'zmatrix '
            output += atom_names[i] + ' '
            if i > 0:
                output += (atom_names[i - 1] + ' ' + str(np.linalg.norm(
                    atom.position_abs - atomlist[i - 1].position_abs) * 1e10) +
                           ' ')

            if i > 1:
                output += atom_names[i - 2] + ' ' + str(
                    math.degrees(self.get_angle(atom, atomlist[i - 1],
                                                atomlist[i - 2]))) + ' '
            if i > 2:
                output += (atom_names[i - 3] + ' ' +
                           str(math.degrees(self.get_dihedral(
                               atom, atomlist[i - 1], atomlist[i - 2], atomlist[i - 3]))) +
                           ' ')
            output += '\n'
        if filename is not None:
            with open(filename, 'w') as myfile:
                myfile.write(output)
            return None

        return output

    def set_constraints_fix_position(self, myatomnumbers):
        """
        Set fixed positions for atoms.

        This function for now only supports CASTEP.

        .. todo::
            * Implement the directions stuff.

        Parameters
        ----------
        myatomnumbers : list of int
            Atomnumbers to fix the position for.
        directions : list of lists
            Up tp three lists containing the directions, in which the atom will
            not be allowed to move. E.g. [[1, 0, 0]] if you want to restrict
            the atoms allowed movement direction to the y/z plane.

        """
        try:
            self.constraints['fix_position']
        except KeyError:
            self.constraints['fix_position'] = {'atomnumbers': [],
                                                'vectors': []}
        self.constraints['fix_position']['atomnumbers'] = myatomnumbers

    def set_nmr_reference(self, mytype, reference):
        """
        Sets the given reference for a atom_type.

        Parameters
        ----------
        reference : float
            The reference shift to add to the isotropic shift.
        type : string
            The atom_type to alter

        """
        # correct for multiple definitions of the reference
        if self.nmr_reference is not None:
            ref = reference - self.nmr_reference
        else:
            ref = reference
        for myatom in self.atoms:
            if myatom.atom_type == mytype:
                myatom.properties['csatensor'].set_reference(ref)

    def set_up_job(self, calcfolder=None, jobname=None,
                   queue_template=None, program='castep'):
        """
        Sets up a job for the cell.

        Allows the user to set up
        a G09 or CASTEP calculation directly from a `Cell` instance.
        All necessary files are automatically created in the appropriate
        folders, ready to be handled by local or queued execution.
        The function can also be used to just 'update' the job, for example
        to rewrite the queue submission file without touching other files, if
        you wish to run the job on another machine.


        Parameters
        ----------
        jobname : string, optional
            The name of the job to be run. Defaults to the `Cell.cellname`
            attribute of the parent cell.
        jobname_q : string
            The name of the job displayed upon ``qstat``.
        calcfolder : string, optional
            The folder in which the calculation is performed in. Comes in
            handy if you start the calculation handing over a `Cell`
            instance and a *param* dictionary, or if you start multiple jobs
            at once, like in a convergence or scan job. Defaults to the
            `Cell.foldername` attribute of the parent cell.
        program : {'gaussian', 'castep'}
            The program to use. Defaults to 'gaussian'.
        queue_template : str
            If supplied, we write a queue file for this
            calculation. We replace *[[input]]* with the input file and
            *[[jobname]]* with the jobname (optional).

        Examples
        --------
        Assume you want to set up a geometry optimisation with a NMR
        calculation following using G09.
        You would first set up a cell. Optionally, you can change some of
        the parameters, or specify a multi-step job. ::

            geom_orig = atomistic.Cell('myexample.log')
            geom_orig.gaussian_params = {
                            'folder': "./CALCS/",
                             'basis_set': "6-31++G**",
                             'jobtype': 'opt',
                             'charge_and_multiplicity': "0 1",
                             'title': "test",
                             'link0_section': "%NProcShared=4",
                             'multi_step_route_section':
                                ["nmr=giao scf=tight"],
                             'multi_step_titles': None,
                             'write_checkfile': 0}

        Now, set up the job and write out the queuing script and the *com*
        file for G09. ::

            geom_orig.set_up_job(calcfolder='CALCS/0/', jobname='scanme_0',
                                 program='gaussian', queue_template=queue_template)

        """
        program = program.lower()
        if program != 'castep' and program != 'gaussian':
            raise ValueError('Program not recognized.')

        if jobname is None:
            jobname = self.cellname if self.cellname is not None else 'myjob'
        if not calcfolder:
            if self.foldername:
                calcfolder = self.foldername
            else:
                calcfolder = './'

        if os.path.isdir(calcfolder) is False and calcfolder != '':
            os.makedirs(calcfolder)
        if program == 'castep':
            self.save_castep(os.path.join(calcfolder, jobname + '.cell'))
            self.save_paramfile(os.path.join(calcfolder, jobname + '.param'))
        if program == 'gaussian':
            self.save_gaussian(os.path.join(calcfolder, jobname + '.com'))

        if queue_template is not None:
            replaces = [('[[input]]', jobname if program == 'castep' else jobname + '.com'),
                        ('[[jobname]]', jobname)]
            for replace in replaces:
                queue_template = queue_template.replace(replace[0], replace[1])
            with open(os.path.join(calcfolder, jobname + '_qsub.sh'),
                      'w') as myfile:
                myfile.write(queue_template)

    def update_lattice_abc(self, abc, fix_abs=False):
        """
        Changes the lattice parameters.

        Changes the lattice parameters with axis lengths and cell angles.
        By default, the atom positions are scaled accordingly. Optionally, the
        absolute atom positions can be fixed.

        Parameters
        ----------
        abc : array_like (6)
            Cell axis lengths and angles in radians.
        fix_abs : bool
            If :const:`True`, the absolute atom positions won't be touched.

        """
        self.lattice_abc = abc

        for myatom in self.atoms:
            myatom.set_basis(self.lattice_cart, fix_abs)

    def update_lattice_cart(self, lattice_cart, fix_abs=False):
        """
        Changes the lattice parameters.

        Changes the lattice parameters with axis vectors. By default, the atom
        positions are scaled accordingly. Optionally, the absolute atom
        positions can be fixed.

        Parameters
        ----------
        lattice_cart : array
            Carthesian lattice vectors [x, y, z].
        fix_abs : bool
            If :const:`True`, the absolute atom positions won't be touched.

        """
        self.lattice_cart = lattice_cart

        for myatom in self.atoms:
            myatom.set_basis(self.lattice_cart, fix_abs)

    @property
    def volume(self):
        """
        Returns the volume of the cell in :math:`m^3`.

        """
        return self.lattice_cart[:, 0] @ np.cross(self.lattice_cart[:, 1],
                                                  self.lattice_cart[:, 2])
