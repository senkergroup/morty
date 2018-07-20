"""
Atom class to be used in Cell.

"""

import os
import re
import numpy as np

from .. import constants
from ..util import fileio


__all__ = ['Atom']


class Atom:
    """
    Defines an Atom and its properties.

    Represents an atom with general information like coordinates,
    information about its coordinate system or properties like type, charge
    and chemical shift tensor.

    Attributes
    ----------
    position_abs : np.ndarray
        The absolute coordinates of the atom.
    position_frac : np.ndarray
        The fractional coordinates of the atom.
    atom_type : str
        The atoms element.
    mass : int
        The Atoms mass number.
    properties : dict
        Stores some of the atoms properties, like charge or chemical shift.
    neighbours : dict
        Holds the neighbours of the atom.

    Notes
    -----
    Properties like the atomic mass, charge, element (named atom_type here)
    and gyromagnetic ratio can be set manually or are set according to default
    values set in :class:`morty.constants` upon instantiation.


    Some common keys for the *properties* attribute are:

    - **gyromagnetic_ratio** (float) – The gyromagnetic ratio of the atom.
    - **chem_group** (string) – The chemical group of the atom. Usually set by
      :class:`morty.atomistic.Cell.det_groups()`.
    - **charge** ({string: float, }) – Holds one pair of key:value for each charge
      scheme they are calculated, e.g 'hirshfeld' or 'mulliken'.
    - **csatensor** (:class:`morty.atomistic.CSATensor`) – The csa tensor
      for the atom.
    - **efgtensor** (:class:`morty.atomistic.EFGTensor`) – The quadrupolar tensor
      for the atom.

    """
    def __init__(self, basis, position_frac=None, position_abs=None,
                 atom_type=None, basis_i=None):
        """
        Instantiates an *Atom*.

        Parameters
        ----------
        atom_type : str
            E.g. 'H' or 'C'.
        position_frac : array
            Position vector given in fractional coordinates.
        position_abs : array_like
            Position vector in absolute coordinates.
        basis : array_like
            Basis in which the position_frac is given/will be calculated.
        properties : dict
            Optional properties for the atom. E.g.
            'csatensor' : :class:`morty.atomistic.CSATensor`.
        nucCharge : str
            Nuclear charge of the Atom
        mass : int
            Atom mass of the atom. By default the atom mass is set to
            the default given in constants.

        """
        self.printed_not_found = {'gmr': False,
                                  'nC': False,
                                  'aT': False,
                                  'mn': False,
                                  'vdw': False}
        if atom_type is not None:
            self.atom_type = atom_type
            self.mass = self.get_mass_number()
            self.nuc_charge = self.get_nuc_charge()

        self._basis = basis
        self._position_frac = position_frac
        self._position_abs = position_abs
        if position_frac is not None:
            self.position_frac = position_frac
        else:
            self.set_position_abs(position_abs, basis_i)
        self.neighbours = []
        self.properties = {}

    def get_group(self, foldername_groupsfile=None, filename_groupsfile=None,
                  ignore_atom_type=None, ignore_chemgroup=None):
        """
        Determine the sort of chemical group the atom belongs to.

        Determine the chemcial group an atom represents according to the
        definitions given in the file *groups*. A default version of this
        file is found in the :class:`morty.external` folder.
        The result is written in ``self.properties['chem_group']`` as well as
        returned.
        Before you run this function, you will have to run
        :class:`morty.atomistic..Cell.det_bonds()` for the class containing
        the atom.

        Notes
        -----
        A more involved syntax allows to define patterns based on already
        known chemical groups. This is rather involved - think real good
        before you use this. ::

            %NUCLEUS
            runningnumber = 0
            type = C
            neighbours = N,N,N_NH:C,C,C
            priority = 1
            name = C_np
            mustbe = 1
            %END_NUCLEUS

        One of the first neighbours is both given by element (N) and by
        chemical group (NH). The chemical group for this nitrogen must be
        known before running *get_group*.

        Parameters
        ----------
        foldername_groupsfile : str
            Path where to search for the *groups* file.
        filename_groupsfile : str
            Path to the *groups* file.
        ignore_atom_type : list
            Ignore a certain *atom_type*, i.e. element. This comes in handy in
            models with e.g. adsorbed atoms you want to disregard.
        ignore_chemgroup : list
            Ignore a certain chemical group. This comes in handy in models with
            adsorbed molecules.
            WARNING: This does NOT account for a recursive problem, where a
            first chemical group ignores a second chemical group which has in
            turn to ignore the first one to be identified.

        Returns
        -------
            match : string
                The match for this Atom.

        """
        groupsconfig = fileio.SsParser()
        if filename_groupsfile is not None:
            groupsconfig.read(filename_groupsfile)
        elif foldername_groupsfile is not None:
            groupsconfig.read(os.path.join(foldername_groupsfile, 'groups'))
        else:
            groupsconfig.read('./groups')
        # delimiters used in the groups file for
        # $: denotes a neighbours chem_group
        # &: denotes one of a neighbours chem_group_additional's
        # =: denotes one of a neighbours chem_group_additional's which has
        #    to match the central atoms one
        delimiters = ['\$', '\&', '\=']
        foundmatch = []
        for i, nucleus_conf in enumerate(groupsconfig.blocks['nucleus']):
            if nucleus_conf['type'] == self.atom_type:
                neighbour_definition = [
                    orderNeighbours.split(',') for
                    orderNeighbours in nucleus_conf['neighbours'].split(':')]
                is_group = True
                if 'mustbe' in nucleus_conf and nucleus_conf['mustbe'] == 1:
                    for neighbour_order, myneighbour in enumerate(neighbour_definition):
                        # check_chemgroup = False
                        # for x in neighbour_definition[neighbour_order]:
                        #     if x.partition('_')[1]:
                        #         check_chemgroup = True
                        # easier way to do this? well, a better one...
                        # not correct. sets are not suited.
                        if ignore_atom_type is not None:
                            ns = [x for x in self.get_neighbours(
                                neighbour_order + 1)
                                  if x.atom_type not in ignore_atom_type]
                        elif ignore_chemgroup is not None:
                            ns = [x for x in self.get_neighbours(
                                neighbour_order + 1)
                                  if x.get_group()['name']
                                  not in ignore_chemgroup]
                        elif 'ignore_atomtype' in nucleus_conf:
                            ns = [x for x in self.get_neighbours(
                                neighbour_order + 1)
                                  if x.atom_type not in
                                  nucleus_conf['ignore_atomtype']]
                        else:
                            ns = [x for x in
                                  self.get_neighbours(neighbour_order + 1)]
                        if len(ns) != len(myneighbour):
                            is_group = False
                            break
                        else:
                            chekker = len(myneighbour)
                            for x in sorted(myneighbour, key=lambda x:
                                            len(re.split('\&|\$|\=', x)),
                                            reverse=True):
                                x_split = [x]
                                for d in delimiters:
                                    x_split = [m for n in x_split
                                               for m in re.split(r'(' + d +
                                                                 '.*?(?=' + d +
                                                                 '|$))', n)
                                               if m]
                                for at in ns:
                                    match = [1, 0]
                                    if x_split[0] == at.atom_type:
                                        match[1] += 1
                                    # check for chemgroup
                                    if '$' in x:
                                        match[0] += 1
                                        if ('chem_group' in
                                                at.properties.keys() and
                                                at.properties['chem_group'] ==
                                                [x for x in x_split
                                                 if '$' in x][0][1:]):
                                            match[1] += 1
                                    # check for additional chemgroup
                                    if '&' in x:
                                        match[0] += len(x.split('&')) - 1
                                        if ('chem_group_additional' in
                                                at.properties.keys()):
                                            for add in [x for x in x_split
                                                        if '&' in x]:
                                                if (add[1:] in at.properties[
                                                        'chem_group_additional'
                                                ]):
                                                    match[1] += 1
                                    # check for common additional chemgroup
                                    if '=' in x:
                                        match[0] += len(x.split('=')) - 1
                                        if ('chem_group_additional' in
                                                at.properties.keys()):
                                            for equ in [x for x in x_split
                                                        if '=' in x]:
                                                if (equ[-1] != '*' and equ[1:] in
                                                        at.properties['chem_group_additional'] and
                                                        equ[1:] in self.properties[
                                                            'chem_group_additional']):
                                                    match[1] += 1
                                                elif (equ[-1] == '*' and
                                                      (True in
                                                       [True if equ[1:].replace('*', '') in
                                                        y else False for y in
                                                        at.properties['chem_group_additional']
                                                       ]) and
                                                      (True in
                                                       [True if equ[1:].replace('*', '') in
                                                        y else False for y in
                                                        self.properties['chem_group_additional']
                                                       ])):
                                                    if ([self.properties['chem_group_additional']
                                                         if equ[1:].replace('*', '')
                                                         in y else False for y in
                                                         self.properties[
                                                             'chem_group_additional']
                                                        ][0] == [
                                                            at.properties['chem_group_additional']
                                                            if equ[1:].replace('*', '')
                                                            in y else False for y in
                                                            at.properties[
                                                                'chem_group_additional']
                                                            ][0]):
                                                        match[1] += 1

                                    if match[0] == match[1]:
                                        ns.remove(at)
                                        chekker -= 1
                                        break

                            if not len(ns) == chekker == 0:
                                is_group = False
                else:
                    for j, neighbour_order in enumerate(neighbour_definition):
                        if (not set(neighbour_order)
                                .issubset([x.atom_type for x in
                                           self.get_neighbours(j + 1)])):
                            is_group = False

                if is_group is True:
                    foundmatch.append(groupsconfig.blocks['nucleus'][i])
        match = sorted(foundmatch,
                       key=lambda fm: fm['priority'])
        try:
            self.properties['chem_group'] = match[-1]['name']
            return match[-1]
        except IndexError:
            self.properties['chem_group'] = None

    def get_gyromagnetic_ratio(self):
        """
        Get the gyromagnetic ratio.

        The gyromagnetic ratio is retrieved from
        :class:`morty.constants`, using the *mass_number*.

        Returns
        -------
        gyromagnetic_ratio : float
            The gyromagnetic ratio of the atom.

        """
        try:
            self.properties['gyromagnetic_ratio'] = \
                constants.GYROMAGNETIC_RATIOS[
                    str(self.mass) + self.atom_type]

            return constants.GYROMAGNETIC_RATIOS[str(self.mass) + self.atom_type]
        except KeyError:
            if self.printed_not_found['gmr'] is False:
                self.printed_not_found['gmr'] = True
                return None

    def get_mass_number(self):
        """
        Get the mass number of the atom.

        The mass number is retrieved from
        :class:`morty.constants`, using the *atom_type*.

        Returns
        -------
        mass_number : int
            The mass number of the atom.

        """
        try:
            return constants.NMR_MASSNUMBERS[self.atom_type]
        except KeyError:
            return None

    def get_neighbours(self, order):
        """
        Determines the neighbours of the atom.

        Determines the neighbours of the atom up to the defined *order*.
        Above the first order, only neighbours covalently bound to atoms in the
        lower shell are considered.

        Notes
        -----
        The first neighbours of the atom have to be known before this
        function is called, i.e. you have to call
        :class:`morty.atomistic.Cell.det_bonds()` before.

        Parameters
        ----------
        order : int
            Defines up to which shell neighbours should be searched for.
            If set to two, the next neighbours would be determined as well as
            the neighbours of these neighbours be set as second neighbours for
            the atom.

        """
        # iterate over bond orders, getting new atoms-to-check-for-neighbours
        # tocheck[] each time
        tocheck = []
        for neighbour in self.neighbours:
            tocheck.append(neighbour)
        # remember atoms that are already neighbours in a lower order: else
        # we're going backwards starting with the third order
        lowerordercontains = [neighbour['atom'] for neighbour in tocheck]

        # loop over coordination spheres
        for my_depth in range(order):
            # if our loop is at the correct order, return the neighbours
            if my_depth == order - 1:
                return [neighbour['atom'] for neighbour in tocheck]
            else:
                newtocheck = []
                for existing_neighbour in tocheck:
                    for newneighbour in existing_neighbour['atom'].neighbours:
                        # avoid duplicates
                        if (newneighbour['atom'] not in newtocheck and
                                newneighbour['atom'] not in
                                lowerordercontains and
                                existing_neighbour['covalent'] == 1 and
                                all([my_depth == 0, newneighbour['atom'] ==
                                     self]) is False):
                            newtocheck.append(newneighbour)
                            lowerordercontains.append(newneighbour['atom'])
                tocheck = newtocheck
        return None

    def get_nuc_charge(self):
        """
        Get the nuclear charge of the atom.

        The nuclear charge is read in e.g. using
        :class:`morty.atomistic.Cell.load_castep_charges()`.

        Returns
        -------
        nuc_charge : float
            The nuclear charge of the atom.

        """
        try:
            self.nuc_charge = constants.NUC_CHARGES[self.atom_type]
            return constants.NUC_CHARGES[self.atom_type]
        except KeyError:
            return None

    def get_vdw_radius(self):
        """
        Get the Van der Waals radius of the atom.

        The VdW radius is retrieved from
        :class:`morty.constants`, using the `atom_type`.

        Returns
        -------
        vdw_radius : float
            The VdW radius of the atom.

        """
        try:
            return constants.VDW_RADII[self.atom_type]
        except KeyError:
            if self.printed_not_found['vdw'] is False:
                self.printed_not_found['vdw'] = True
                return None

    def is_neighbour(self, order, otheratom):
        """
        Check if another atom is a neighbour of the atom.

        Parameters
        ----------
        order : int
            The order which should be checked.
        otheratom : :class:`morty.atomistic.Atom`
            The other atom to check for neighbourhood to the atom.

        Returns
        -------
        is_neighbour : bool
            :const:`True` if the `otheratom` is a neighbour of `order` to the atom.

        """
        neighbours = self.get_neighbours(order)
        return neighbours is not None and otheratom in neighbours

    def set_basis(self, basis, fix_abs=True):
        """
        Change the basis of the atom.

        Change the basis of the atom and keep either the absolute or the
        fractional coordinates unchanged.

        Parameters
        ----------
        basis : np.ndarray
            The new basis.
        fix_abs : bool
            If :const:`True`, the absolute coordinates are retained, otherwise the
            fractional coordinates.

        """
        self._basis = basis
        if fix_abs is True:
            self.position_abs = self.position_abs
        else:
            self.position_frac = self.position_frac

    def set_neighbour(self, distance, neighbour, covalent,
                      neighbour_translated):
        """
        Set a neighbour of the atom.

        Set a new neighbour of the atom, including the full specification.

        Parameters
        ----------
        distance : float
            The distance of the atom to the neighbour, in Å.
        neighbour : :class:`morty.atomistic.Atom`
            The neighbours instance.
        covalent : bool
            If the bond is to be handled as covalent, meaning the neighbours
            atoms are considered as second neighbours of the atom.
        neighbour_translated : np.array
            The translated coordinates of the neighbour. This is necessary,
            if the neighbour is found in a translated unitcell.

        """
        double = False
        for old_neighbour in self.neighbours:
            if old_neighbour['atom'] == neighbour:
                double = True
        if double is False:
            self.neighbours.append({'atom': neighbour, 'distance': distance,
                                    'covalent': covalent,
                                    'atom_translated_coords':
                                    neighbour_translated})

    @property
    def position_frac(self):
        """
        The fractional coordinates of the atom.
        :type: np.ndarray

        """
        return self._position_frac

    @property
    def position_abs(self):
        """
        The absolute coordinates of the atom.
        :type: np.ndarray

        """
        return self._position_abs

    @position_abs.setter
    def position_abs(self, val):
        self.set_position_abs(val)

    def set_position_abs(self, my_position_abs, my_basis_i=None):
        """
        Sets the coordinates of the atom in absolute coordinates.

        You do not need to use this, you can just set `position_abs`.
        This is a helper function which allows to provide a inverted
        basis, to speed up the calculation of the fractional coordinates.
        This is used when loading large trajectories, because the
        inversion has to be performed only once per cell in that case.

        Parameters
        ----------
        my_position_abs : array_like
            The absolute position of the atom in Å.
        my_basis_i : np.ndarray
            The inverted basis of the atom. Speeds up the calculation of the
            fractional coordinates.

        """
        self._position_abs = (my_position_abs if isinstance(my_position_abs,
                                                            np.ndarray)
                              else np.array(my_position_abs))
        if my_basis_i is not None:
            self._position_frac = self._position_abs.dot(my_basis_i)
        else:
            self._position_frac = self._position_abs.dot(
                np.linalg.inv(self._basis))

    @position_frac.setter
    def position_frac(self, my_position_frac):
        """
        Sets the coordinates of the atom in fractional coordinates.

        The absolute coordinates are adapted accordingly.

        Parameters
        ----------
        my_position_frac : np.ndarray
            The fractional position of the atom in Å.

        """
        self._position_frac = (my_position_frac if isinstance(my_position_frac,
                                                              np.ndarray)
                               else np.array(my_position_frac))
        self._position_abs = self._position_frac.dot(np.asarray(self._basis))
