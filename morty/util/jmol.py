"""
Integrate JSMol.



"""

import os
import shutil
import platform
import itertools
import IPython.core.display
from IPython.core.display import HTML
import numpy as np
from .. import constants


__all__ = ['JmolHandler']


COLOR_TABLE = {'O': 'red', 'C': 'grey', 'N': 'blue', 'F': 'green'}


class JmolHandler():
    """
    Allows to display a :class:`morty.modeling.Cell` instance inside the IPython Notebook.

    Includes JSmol inside the Notebook to display any structure that can be
    loaded by :class:`morty.modeling.Cell`.


    Notes
    -----
    The current version is running locally. The JSmol frame is run via the
    tornado server your Ipython notebook is using.

    Examples
    --------

    **Chemical Groups**

    Let us assume you want to display a cell with the carbons labelled
    with their chemical group assignment: ::

        mycell = morty.modeling.Cell('somecell.cell')
        mycell.det_bonds()
        mycell.det_groups()
        jmol1 = JmolHandler(cell=mycell, labels=['chemgroup', 'C'])
        jmol1.show()

    **Chemical shifts**

    Let us assume you have your (periodic, CASTEP-treated) structure in a
    file 'blubb.cell', and the corresponding nmr calculation in
    'blubb.magres'.
    Let us also assume that you have the deep wish to suppress all bonds
    Jmol draws for lithium and draw a supercell.
    First, you would want to load these files: ::

        myblubb = morty.modeling.Cell('blubb.magres')

    Now you can use a Jmol applet to plot the structure and the calculated
    tensors: ::

        myjmol = morty.util.JmolHandler(cell=myblubb, csa=True,
                                        csa_individual_scale=True)
        myjmol.show(add_script="select lithium; connect delete; select 0;",
                    load='{2 2 1}')

    """
    def __init__(self, cell=None, trajectory=None, labels=None, csa=False,
                 csa_max=1, csa_individual_scale=False, tensor_types=None,
                 tensor_atomnos=None, csa_tensor_include_isotropic=False,
                 just_molecule=False, efg=False, jmolfolder='JSmol', load='',
                 add_script='', pre_script='background white;'):
        """
        The Jmol instance to hold most of the options to display.

        Parameters
        ----------
        cell: :class:`morty.modeling.Cell`
            Cell to be displayed.
        trajectory: :class:`morty.modeling.Trajectory`
            Trajectory to be displayed.
        labels : tuple (str, list)
            You can plot various properties on the atoms. The first argument
            gives the property, the second the ``atom_type`` (s) to plot for.
            May be one of:

            - **default** – Plots the standard labels of Jmol.
            - **chemgroup** – Plots the chemical group.
            - **shifts** – Plots the chemical shift.
            - **charges:mulliken**, **charges:hirshfeld** – Displays charges of
              the specified type.
            - **castep_bondorders** (dict) – Display the bond orders extracted
              via :class:`morty.modeling.Cell.load_castep_bondorders()`

            The second argument for *labels* hands over optional arguments:

            - **atom_types** (list of str) – List of atomtypes to include.
            - **min_bondorder** (float) – Minimum bond order.
            - **shift_labels** (array) – Jmol often draws the labels awkwardly.
              This is a 3d-vector to shift the labels where they belong.
            - **translated_labels** (dict) – with keys

                - *shell* (list)
                - *point* (list)
                - *dist* (float)

                Translate the labels also to adjacent cells. If set, only
                display the ones withinx *dist* of *point*.

        csa : bool
            Set to :const:`True` to display the csa tensors in the applet.
        csa_max : int
            Size of the biggest CSA tensor in Angstrom.
        csa_individual_scale : bool
            Scale each atom type individually to csa_max.
        tensor_types : array_like
            Atom types to display CSA/EFG tensors for, e.g. ['H', 'C'].
        tensor_atomnos : array_like
            Atom number to display CSA/EFG tensors for.
        csa_tensor_include_isotropic : bool
            Include the isotropic part of the CSA tensor.
        efg : bool
            Set to :const:`True` to display the EFG tensors in the applet.
        load: str
            The supercell to load in the applet. Defaults to '', which
            displays no supercell.
        add_script : str
            Additional Jmol script for display options. Executed after
            creating the geometry and other elements.
        pre_script : str
            Addition Jmol script to execute before all other commands.
            Seldomly necessary, e.g. for setting general styles for echos.

        """
        self.cell = cell
        self.traj = trajectory
        self.labels = labels
        self.csa = csa
        self.efg = efg
        self.jmolfolder = jmolfolder
        self.csa_max = csa_max
        self.csa_individual_scale = csa_individual_scale
        self.tensor_types = tensor_types
        self.tensor_atomnos = tensor_atomnos
        self.tensor_include_isotropic = csa_tensor_include_isotropic
        self.just_molecule = just_molecule
        self.load = load
        self.add_script = add_script
        self.pre_script = pre_script
        self.defaults = 'color background darkgrey;\n'

    def construct_tensor(self, whichtensor='csatensor', atom_numbers=None):
        """
        Constructs the Jmol ellipsoids from the atoms tensors.

        Parameters
        ----------
        whichtensor : str
            Which tensor to plot. Must be one of 'csatensor' or
            'efgtensor'. Used to pick properties of an atom.
        atom_numbers : list
            List of atom numbers to show.

        """
        mytensorscript = str()
        atoms = self.cell.get_atoms(atom_numbers)

        # First, determine max Values for the CSA scaling.
        if self.csa_individual_scale is True:
            my_max = {}
            for atom in atoms:
                if (self.tensor_include_isotropic is True or
                        whichtensor == 'efgtensor'):
                    my_tensor = atom.properties[whichtensor].tensor
                else:
                    my_tensor = atom.properties[whichtensor
                                               ].get_anisotropic_tensor()
                if np.max(np.abs(my_tensor)) >= my_max.get(atom.atom_type, 0):
                    my_max[atom.atom_type] = np.max(np.abs(my_tensor))
        else:
            my_max = 0
            for atom in atoms:
                if (self.tensor_include_isotropic is True or
                        whichtensor == 'efgtensor'):
                    my_tensor = atom.properties[whichtensor].tensor
                else:
                    my_tensor = atom.properties[whichtensor
                                               ].get_anisotropic_tensor()
                if np.max(np.abs(my_tensor)) >= my_max:
                    my_max = np.max(np.abs(my_tensor))

        i = 1
        for atom in atoms:
            if (self.tensor_include_isotropic is True or
                    whichtensor == 'efgtensor'):
                my_tensor = atom.properties[whichtensor].tensor
            else:
                my_tensor = atom.properties[whichtensor
                                           ].get_anisotropic_tensor()
            # This is some hack to create 100% orthogonal vectors. Seems like
            # jmol had some problems if the differed even by very small values.
            my_tensor3 = np.matrix([[None] * 3] * 3)
            if self.csa_individual_scale is True:
                eigenvals, eigenvecs = np.linalg.eigh(
                    my_tensor / my_max[atom.atom_type])
            else:
                eigenvals, eigenvecs = np.linalg.eigh(my_tensor / my_max)
                eigenvecs = np.array(eigenvecs)
            my_tensor3[0] = (eigenvals[0] * eigenvecs[:, 0])
            my_tensor3[1] = (eigenvals[1] * eigenvecs[:, 1])
            my_tensor3[2] = (eigenvals[2] * eigenvecs[:, 2])
            mytensorscript += 'ellipsoid ID ' + str(i) + ' AXES {'
            mytensorscript += (str(my_tensor3[0, 0]) + ' ')
            mytensorscript += (str(my_tensor3[0, 1]) + ' ')
            mytensorscript += (str(my_tensor3[0, 2]) + '} {')
            mytensorscript += (str(my_tensor3[1, 0]) + ' ')
            mytensorscript += (str(my_tensor3[1, 1]) + ' ')
            mytensorscript += (str(my_tensor3[1, 2]) + '} {')
            mytensorscript += (str(my_tensor3[2, 0]) + ' ')
            mytensorscript += (str(my_tensor3[2, 1]) + ' ')
            mytensorscript += (str(my_tensor3[2, 2]) + '};\n')
            if self.csa_individual_scale is True:
                mytensorscript += ('ellipsoid ID ' + str(i) + ' SCALE ' +
                                   str(self.csa_max) +
                                   ';\n')
            else:
                mytensorscript += ('ellipsoid ID ' + str(i) + ' SCALE ' +
                                   str(self.csa_max) + ';\n')
            mytensorscript += ('ellipsoid ID ' + str(i) + ' CENTER ' +
                               '{atomno=' + str(atom_numbers[i - 1]) + '};\n')
            mytensorscript += ('ellipsoid ID ' + str(i) + ' COLOR ' +
                               'translucent .2 ' +
                               COLOR_TABLE.get(atom.atom_type, 'grey') + ';\n')
            i += 1

        return mytensorscript

    def get_script(self, load=''):
        """
        Returns the script for displaying the structure in Jmol.

        This contains everything that is needed to show the structure, that
        is shown in the applet. It includes the structure data and can be
        directly loaded into Jmol.

        """
        if load == '':
            load = self.load
        myscript = self.defaults
        if self.traj is not None:
            traj_load = list(["load data 'model' \n" +
                              self.traj[0].get_cell_format() +
                              "end 'model'"] +
                             [str("load data 'append model' \n" +
                                  self.traj[i].get_cell_format() +
                                  "end 'append model'")
                              for i in range(1, len(self.traj))])
            myscript += str(" " + load + "\n"
                           ).join(traj_load) + " " + load + "\n"
        elif self.cell is not None:
            myscript += ("load data 'model'\n" +
                         self.cell.get_cell_format() +
                         "end 'model' " + load + "\n")
        else:
            raise RuntimeError('No structure found.')
        myscript += self.pre_script
        if self.labels is not None and self.traj is None:
            if self.labels == 'default' or self.labels == 'atomnumber':
                myscript += 'label %[atomname]\n'
            elif self.labels[0] == 'default' or self.labels[0] == 'atomnumber':
                myscript += 'select 0;\n'
                for selected in self.labels[1]:
                    myscript += str(
                        'select add ' +
                        constants.ATOM_TYPES[selected] + ";\n")
                myscript += 'label %[atomname];\nselect 0;'
            elif (self.labels == 'chemgroup' or
                  self.labels[0] == 'chemgroup' or
                  self.labels == 'chem_group' or
                  self.labels[0] == 'chem_group'):
                myscript += 'select 0;\n'
                for atom_number in range(len(self.cell.atoms)):
                    if (self.labels == 'chemgroup' or
                            self.labels == 'chem_group' or
                            self.cell.atoms[
                                atom_number].atom_type in self.labels[1]):
                        myscript += 'select @' + str(atom_number + 1) + '; '
                        if ('atomnumber' in self.labels or
                                'atomnumber' in self.labels[0]):
                            myscript += 'label %[atomname] - '
                        else:
                            myscript += 'label '
                        myscript += str(
                            self.cell.atoms[atom_number].properties[
                                'chem_group']) + ';\n'
                myscript += 'select 0;\n'
            elif 'charge' in self.labels or 'charge' in self.labels[0]:
                myscript += 'select 0;\n'
                for atom_number in range(len(self.cell)):
                    try:
                        chargetype = self.labels[0].split('charge:')[1].split(',')[0]
                    except IndexError:
                        chargetype = self.labels.split('charge:')[1].split(',')[0]
                    if ('charge' in self.labels or
                            self.cell.get_atom(atom_number + 1).atom_type in
                            self.labels[1]):
                        myscript += 'select @' + str(atom_number + 1) + '; '
                        myscript += 'label '
                        if ('atomnumber' in self.labels or
                                'atomnumber' in self.labels[0]):
                            myscript += '%[atomname]/'
                        myscript += str(self.cell.get_atom(atom_number + 1)
                                        .properties['charge'][chargetype]) + ';\n'
            elif (self.labels == 'shift' or self.labels[0] == 'shift' or
                  self.labels == 'shifts' or self.labels[0] == 'shifts'):
                myscript += 'select 0;\n'
                for atom_number in range(len(self.cell.atoms)):
                    if (self.labels == 'shift' or
                            self.cell.get_atom(atom_number + 1).atom_type in
                            self.labels[1]):
                        myscript += 'select @' + str(atom_number + 1) + '; '
                        myscript += 'label '
                        if ('atomnumber' in self.labels or
                                'atomnumber' in self.labels[0]):
                            myscript += '%[atomname] - '
                        myscript += str(round(self.cell.get_atom(
                            atom_number + 1).properties['csatensor'
                                                       ].hms[0], 2)) + '\n'
                myscript += 'select 0\n'
            elif (self.labels == 'castep_bondorders' or
                  self.labels[0] == 'castep_bondorders'):
                try:
                    atomtypes = self.labels[1]['atom_types']
                except (KeyError, IndexError):
                    atomtypes = 'all'
                try:
                    min_bondorder = self.labels[1]['min_bondorder']
                except (KeyError, IndexError):
                    min_bondorder = None
                try:
                    shift_echo = self.labels[1]['shift_labels']
                except (KeyError, IndexError):
                    shift_echo = [0.0, 0.0, 0.0]
                translated_labels = {}
                try:
                    translated_labels['shell'] = self.labels[1]['translated_labels']['shell']
                except (KeyError, IndexError):
                    translated_labels['shell'] = [1, 1, 1]
                try:
                    translated_labels['point'] = self.labels[1]['translated_labels']['point']
                except (KeyError, IndexError):
                    translated_labels['point'] = None
                try:
                    translated_labels['within'] = self.labels[1]['translated_labels']['within']
                except (KeyError, IndexError):
                    translated_labels['within'] = None
                j = 0
                for i in range(len(self.cell.properties['castep_bondorders'])):
                    atom1, atom2, bondorder = self.cell.properties['castep_bondorders'][i]
                    if ((bondorder >= min_bondorder or min_bondorder is None) and
                            ((self.cell.get_atom(atom1).atom_type in atomtypes and
                              self.cell.get_atom(atom2).atom_type in atomtypes) or
                             atomtypes == 'all')):
                        location = (self.cell.get_atom(atom1).position_abs +
                                    self.cell.get_shortest_distance(
                                        self.cell.get_atom(atom1),
                                        self.cell.get_atom(atom2),
                                        returnnewatom2=True)[1].position_abs) / 2
                        for x, y, z in itertools.product(range(translated_labels['shell'][0]),
                                                         range(translated_labels['shell'][1]),
                                                         range(translated_labels['shell'][2])):
                            j += 1
                            echopos = location + np.asarray(np.matrix(
                                self.cell.lattice_cart).T * np.array([x, y, z]).astype(
                                    np.float).reshape(3, 1)).reshape(1, 3)[0]
                            if (np.linalg.norm(echopos - translated_labels['point']) <
                                    translated_labels['within'] or
                                    translated_labels['point'] is None):
                                myscript += str('set echo bo' + str(j) + ' {' + ' '.join(
                                    [str(x) for x in (echopos + np.array(shift_echo)) * 1e10]) +
                                                '}; echo ' + str(bondorder) + ';')
                myscript += '\n'

        if self.traj is None:
            if self.csa is True:
                myatoms = []
                if self.tensor_types is not None:
                    for atom in self.cell.get_atoms(names=self.tensor_atomnos):
                        if atom.atom_type in self.tensor_types:
                            myatoms.append(self.cell.atoms.index(atom) + 1)
                else:
                    if self.tensor_atomnos is not None:
                        myatoms = self.tensor_atomnos
                    else:
                        myatoms = list(range(1, len(self.cell) + 1))
                myscript += self.construct_tensor('csatensor', myatoms)

            if self.efg is True:
                myatoms = []
                if self.tensor_types is not None:
                    for atom in self.cell.get_atoms(names=self.tensor_atomnos):
                        if atom.atom_type in self.tensor_types:
                            myatoms.append(self.cell.atoms.index(atom) + 1)
                else:
                    if self.tensor_atomnos is not None:
                        myatoms = self.tensor_atomnos
                    else:
                        myatoms = list(range(1, len(self.cell) + 1))
                myscript += self.construct_tensor('efgtensor', myatoms)

        if self.just_molecule is True:
            myscript += 'axes off; unitcell off;'
            myscript += 'select all; center selected; select 0;\n'

        return myscript

    def show(self, inline=True, pre_script='', add_script='', load='',
             width=480, height=320):
        """
        Displays the Jmol applet inside a Notebook.

        Parameters
        ----------
        script : str
            If `script` is set, the script isn't created automatically and your
            input is used.
        inline : bool
            If :const:`True`, displays a JSmol applet within the IPython Notebook.
            If :const:`False`, opens a Jmol window with the model.
        add_sript: str
            A script which is simply added to the default script.
        load: str
            The supercell to load in the applet. Defaults to '{1 1 1}', which
            displays no supercell.
        width : int
            The width of the applet.
        height : int
            The height of the applet.

        """
        if load == '':
            load = self.load
        if add_script != '':
            self.add_script = add_script
        if pre_script != '':
            self.pre_script = pre_script

        if ('jsmol' not in os.listdir()) or (not os.path.isdir('jsmol')):
            # next statements only work under win when
            # user is administrator, therefore we copy
            if platform.system() != 'Windows':
                if os.path.islink('./jsmol'):
                    os.remove('./jsmol')
                os.symlink(os.path.join(
                    os.path.dirname(__file__), '../external/jsmol'), 'jsmol')
            else:
                if not os.path.isdir('./jsmol'):
                    shutil.copytree(os.path.join(
                        os.path.dirname(__file__), '../external/jsmol'),
                                    'jsmol')
        myscript = str(
            self.get_script(load=load) + ';\n' +
            self.add_script + '\n')
        if inline is True:
            srcdoc = str(open(
                os.path.join(
                    os.path.dirname(__file__),
                    '../external/default_jsmol_applet.html'),
                'r').read().replace('\n', '').replace('"', '&quot;'))
            htmldata = str('<!DOCTYPE html>\n<html>\n  <body>\n' +
                           '  <iframe srcdoc="' +
                           str(srcdoc.replace('[[myjmolscript]]', '&quot;' +
                                              myscript.replace('\n', '\\n') +
                                              '&quot; ;')) + '" ' +
                           'width="' + str(1.1 * width) + '"; ' +
                           'height="' + str(1.1 * height) + '"; ' +
                           'frameBorder="0"></iframe>\n' +
                           '</body>\n</html>')
            IPython.core.display.display(HTML(htmldata))
        else:
            self.save_jmol_script(jmolfilename='tmp.jmol', load=load)
            os.system('jmol -s tmp.jmol')

    def save_jmol_script(self, jmolfilename='default.jmol', load=''):
        """
        Saves a jmol script of the current *JmolHandler* instance.

        A Jmol script is saved, respecting all settings and instruction set
        for the *JmolHandler* at hand, including custom scripts.

        """
        if load == '':
            load = self.load
        with open(jmolfilename, 'w') as myjmolfile:
            myjmolfile.write(str(
                self.get_script(load=load) + ";\n" +
                self.add_script + "\n"))
