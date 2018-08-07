"""
The Cellmodeller class provides tools for automated modelling.


"""

import inspect
import types
import numbers
import copy
import numpy as np
import os.path

from . import Cell

__all__ = ['Cell', 'CellModeller']

class CellModeller:
    """
    Modify the Cell using various transformations.

    Allows to apply various transformations to the cell. Provides an interface
    for optimizing these transformations by varying the paramters and exporting
    modeling.Spinsystem()s. The definition of the transformations can be
    found in the :class:`morty.modeling.Cell` class definition.

    Examples
    --------
    Usually, you would initialize this class with a starting structure, ::

        cell_modeller = CellModeller(mycell)

    and define certain transformations, which returns the id of the
    transformation, e.g. a rotation about a user-defined axis. ::

        id_ rot = cell_modeller.add_transformation('mod_rotate_with_fixed_axis',
            [atoms, point, vector])

    Now you can use `do_transformations` or `do_all_transformations` to create
    new structures, e.g.  ::

        newCell = cell_modeller.do_transformations([id_rot],
            [[np.radians(20)]])

    Parameters
    ----------
    cell : :class:`morty.modeling.Cell`
        The cell instance, with which the transformations start.

    Notes
    -----
    The *CellModeller* can handle two types of functions: The ones listed in
    its property *_function_mapping*, and ones handed over by the user.
    The advantage of *_function_mapping* is, that the functions used can be
    tested for correct usage. So, whenever possible, one should use these
    functions. If you feel pressed to use other functions, just keep in mind
    the following:

    * The last argument is the dynamic one.
    * There will be no default setting for sane boundaries, naturally.
    * Up until now, no return values are handled.

    Available transformations in *_function_mapping*:

    `mod_rotate_with_fixed_axis`
        Rotates a given set of atoms around a static axis. The axis is defined
        by a vector and its foot.

        Static arguments:
            * atoms : array of int
                The numbers of the atoms to be rotated.
            * axis_point : vector
                The foot defining the axis in conjunction with the vector.
            * axis_vector : vector
                The vector defining the orientation of the axis.

        Dynamic arguments:
            * angle : float (radians)
                The magnitude of the angle to rotate about.

    `mod_rotate_with_dynamic_axis`

        Rotates a given set of atoms around the connection between two atoms.

        Static arguments:
            * atoms : array of int
                The numbers of the atoms to be rotated.
            * atom1 : int
                The first atom defining the rotation axis.
            * atom2 : int
                The second atom defining the rotation axis.

        Dynamic arguments:
            * angle : float (radians)
                The magnitude of the angle to rotate about.

    `mod_move_atoms`

        Moves atoms along a given vector.

        Static arguments:
            * atoms : array of int
                The numbers of the atoms to be dislocated.
            * vector : vector
                The vector along which the movement takes place.
        Dynamic arguments:
            * distance : float
                The magnitude of the vector to move about.

    """
    # (name, function name, number of static parameters, number of dynamic
    # attributes, list of (lower boundary, upper boundary))
    _function_mapping = {'mod_rotate_with_fixed_axis':
                         ('mod_rotate_with_fixed_axis', 3, 1,
                          [(0, 2 * np.pi)]),
                         'mod_rotate_with_dynamic_axis':
                         ('mod_rotate_with_dynamic_axis', 3, 1,
                          [(0, 2 * np.pi)]),
                         'mod_move_atoms':
                         ('mod_move_atoms', 2, 1, [(0, 5e-9)]),
                         'mod_make_nx':
                         ('mod_make_nx', 3, 1, [(-1e-8, 1e-8)])}

    def __init__(self, cell):
        # list of tuples: (name, parameter_set)
        self._transformations = []
        self.cell = cell

        # Raserize parameters
        self._total_nos = 0
        self.number_of_steps = 0
        self._num_consecutive = 0
        self._dimensions = 0
        self._group_transformations = None
        self._boundaries = None
        self.cells = []
        self.total_nos_accepted = 0
        self._accepted_steps = 0
        self._testfunctions = None

    def __iter__(self):
        for cell in self.cells:
            yield cell

    def __getitem__(self, key):
        return self.cells[key]

    def __len__(self):
        return self.total_nos_accepted

    def add_transformation(self, function_name, static_arguments,
                           boundaries=None):
        """
        Add a transformation for later execution.

        Add a transformation and its necessary static arguments to the list of
        transformations, which can be performed via the `do_transformations` or
        `do_all_transformations` methods. For a list of available
        transformations see the class documentation.

        Parameters
        ----------
        function_name : string
            The functionname of the desired transformation. It corresponds to
            transformations being member of the Cell class. They are usually
            denoted by a leading ``mod_`` in their respective name.
            For experts only: you can also hand over a function to be used.
            This function will then be appended to the class instance used.
            Needless to say, this is risky business. Use this function with
            extreme care, and don't come complaining if you break something.
        static_arguments : list
            Holds the static arguments passed to the respective transformation.
            See the `CellModeller` documentation or the one of the
            transformation for details.
        boundaries :
            Define the boundaries for the dynamic argument, which is passed
            upon application of the transformation sceleton defined here.
            The boundaries are used when rasterizing or optimising over the
            transformations. The boundaries are 'inclusive'!. That means, if
            you hand over [0,3], and define 4 as total number of steps late on,
            you will end up with the steps at [0, 1, 2, 3].

        Returns
        -------
        id : int
            ID of the added transformations. That is just a counter, starting
            from 0 for every transformation added. Is needed for
            `do_transformations`.

        """
        # Do we have this transformation?
        if isinstance(function_name, str):
            if function_name.lower() in self._function_mapping.keys():
                # Do we have the right number of static arguments?
                if (len(static_arguments) ==
                        self._function_mapping[function_name.lower()][1]):
                    # Do we have special boundaries?
                    if boundaries is not None:
                        # Do the number of special boundaries match the
                        # arguments?
                        if (len(boundaries) ==
                                self._function_mapping[
                                    function_name.lower()][2]):
                            self._transformations.append((function_name,
                                                          static_arguments,
                                                          boundaries))
                        else:
                            raise ValueError("Number of boundaries doesn't " +
                                             'match number of dynamic ' +
                                             'arguments')
                    # No special boundaries, let's use our predefined ones.
                    else:
                        self._transformations.append(
                            (function_name, static_arguments,
                             self._function_mapping[function_name.lower()][3]))
                    return len(self._transformations) - 1
                else:
                    raise ValueError("Number of static arguments doesn't " +
                                     "match function!")
            raise ValueError('Function not found!')
        else:
            try:
                if (len(static_arguments) ==
                        len(inspect.getfullargspec(function_name)[0]) - 2):
                    self.cell._custom_mod_functions.append(
                        types.MethodType(function_name, self.cell))
                    self._transformations.append((
                        len(self.cell._custom_mod_functions) - 1,
                        static_arguments,
                        boundaries))
                    return len(self._transformations) - 1
                else:
                    raise ValueError("Number of static arguments doesn't " +
                                     "match function definition!")
            except AttributeError:
                raise AttributeError(
                    'Could not use the transformation. ' +
                    'I am fully aware this message is not very ' +
                    'informative, but what can I say....you have been ' +
                    'warned...')

    def do_transformations(self, ids, arguments):
        """
        Performs a series of transformations.

        Performs a series of transformations and returns the transformed cell.
        The id is returned by `add_transformation` or is simply incremented for
        every transformation you add (starting with 0).

        .. todo:: Check our boundaries.

        Parameters
        ----------
        ids : array_like
            List of IDs of the transformations, returned from
            `add_transformation`. Warning: this has to be an array, even
            if it only contains one element.
        arguments : array
            Dynamic arguments of the transformation. See class documentation
            for these. Warning: each dynamic argument _has_ to be handed over
            in the form of an array, even if it only contains one element.

        Returns
        -------
        cell : Cell()
            A cell on which all the given transformations have been performed.

        """
        new_cell = copy.deepcopy(self.cell)
        for i, myid in enumerate(ids):
            if isinstance(self._transformations[myid][0], str):
                if (self._function_mapping[self._transformations[myid][0]][2] ==
                        len(arguments[i])):
                    func = getattr(new_cell,
                                   self._function_mapping[
                                       self._transformations[myid][0]][0])
                    func(*(list(self._transformations[myid][1]) +
                           list(arguments[i])))
                else:
                    raise ValueError("Number of dynamic arguments doesn't " +
                                     'match function!')
            else:
                func = new_cell._custom_mod_functions[
                    self._transformations[myid][0]]
                func(*(list(self._transformations[myid][1]) +
                       list(arguments[i])))
        return new_cell

    def get_transformations_boundaries(self):
        """
        Returns boundaries of the arguments available for the transformations.

        Returns
        -------
        boundaries : list of list of tuples (lower, upper)
            Every item in the list represents a transformation, every item in
            this list represents a dynamic parameter of the transformation and
            is a tuple of (lower boundary, upper boundary).

        """
        boundaries = [None] * len(self._transformations)
        for i, transformation in enumerate(self._transformations):
            boundaries[i] = transformation[2]
        return boundaries

    def rasterize_setup(self, number_of_steps, group_transformations=None, testfunctions=None):
        """
        Set parameters to rasterize the transformations.

        Parameters
        ----------
        number_of_steps : array of int/int
            Number of steps that are rasterized for each parameter.
        group_transformations : array of arrays
            If several transformations are to be performed simultaneously, give
            all transformation groups here. If you have 3 transformations, and
            want the first two to be applied simultaneously, you should set
            ``group_transformations=((0, 1), (2,))``.
        testfunctions : list of [name/function, params, bool]
            Hands over a list of testfunction to test the steps against. If the
            step yields 'True' for each testfuntion, it is accepted.
            name: The name of a function (has to be a method of
            :class:`morty.modeling.Cell`) or the function itself to serve as
            acceptance trigger for the steps. If a function is given, the currently
            tested step is handed over as the first arument, then the params.
            params: The parameters are handed over as a list, which is unpacked
            and handed over to the function.
            bool: Steps yielding bool via this function are accepted.

        Examples
        --------
        If for example you aim for
        - using G09
        - a raster along a hydrogen bond
        - made up by H3 and O4
        - between 3 Angstroms above and below the equilibrium distance
        - in 5 steps
        you can use the rasterizer like that: ::

            geom_orig = modeling.Cell()
            geom_orig.load_gaussian_logfile('EthanolWater.log')
            mycellModeller = modeling.CellModeller(geom_orig)
            mycellModeller.add_transformation('mod_move_atoms',
                [[3, 4], geom_orig.get_atom(3).position_abs -
                 geom_orig.get_atom(4).position_abs],
                boundaries=[(-0.3e-10, 0.3e-10)])
            myRaster = calculations.rasterize_setup([5])
            dft = calculate.DFTCaller(myRaster)
            dft.create_raster(foldername='CALCS', jobname='scanme',
                program=gaussian)

        You will end up with a *CALCS* folder containing one folder for each
        calculation, also holding a *submitusall.sh* file to submit all the queuing
        files.

        """
        self._testfunctions = testfunctions

        self._boundaries = self.optimizer_boundaries()
        if isinstance(number_of_steps, numbers.Number):
            number_of_steps = (number_of_steps, )
        self._group_transformations = group_transformations
        self._dimensions = (len(self._boundaries) if group_transformations is None
                            else len(group_transformations))
        self.number_of_steps = np.asarray(number_of_steps)
        # = Total Number Of Steps.
        self._total_nos = np.prod(self.number_of_steps)
        # Number of same consecutive indexes if we increment the indexes from
        # the right (=> for the last element it's always 1, for the second but
        # last its the number of steps of the last element)
        self._num_consecutive = [1 if (i + 1) == self._dimensions else
                                 np.prod(self.number_of_steps[i + 1:
                                                              self._dimensions])
                                 for i in range(self._dimensions)]

        if ((group_transformations is None and
             len(self.number_of_steps) != len(self._boundaries)) or
                (group_transformations is not None and
                 len(self.number_of_steps) != len(group_transformations))):
            raise ValueError("number_of_steps and number of boundaries don't match!")

        if self._group_transformations is not None:
            new_numberofsteps = [None] * len(self._boundaries)
            new_numconsecutive = [None] * len(self._boundaries)
            for i in range(len(self._boundaries)):
                for j, group in enumerate(self._group_transformations):
                    if i in group:
                        new_numberofsteps[i] = self.number_of_steps[j]
                        new_numconsecutive[i] = self._num_consecutive[j]
                        break
            self.number_of_steps = new_numberofsteps
            self._num_consecutive = new_numconsecutive

        self.total_nos_accepted = 0
        self._accepted_steps = []
        for i in range(self._total_nos):
            if self._rasterize_is_accepted(i):
                self.total_nos_accepted += 1
                self._accepted_steps.append(i)

        self.cells = [None] * self.total_nos_accepted
        for i in range(self.total_nos_accepted):
            self.cells[i] = self.rasterize_get_cell(self._accepted_steps[i])
            self.cells[i].foldername = os.path.join(self.cell.foldername,
                                                    str(i))


    def _rasterize_is_accepted(self, i):
        if self._testfunctions is None:
            return True
        tmp_step = self.optimizer_interface(
            self.rasterize_get_parameters(i))
        acceptit = 0
        for test in self._testfunctions:
            if isinstance(test[0], str):
                if (getattr(tmp_step, test[0])(
                        *(test[1]))[0] is test[2]):
                    acceptit += 1
                else:
                    if test[0](tmp_step, i, *(test[1])) is test[2]:
                        acceptit += 1
        return acceptit == len(self._testfunctions)

    def rasterize_get_parameters(self, n):
        """
        Returns the parameters for index n.

        Parameters for index n, when incrementing the parameters
        one after another (i.e. 0 0, 0 1, 1 0, 1 1). Starts counting at 0.

        Parameters
        ----------
        n : int

        Returns
        -------
        parameters : array

        """
        if n >= self._total_nos or n < 0:
            raise ValueError('Index not valid!')
        return tuple(
            # This gives an integer that can be multiplied with one
            # step of the parameter that we are calculating right now.
            boundary[0] + (np.floor(n / self._num_consecutive[i]) % self.number_of_steps[i]) *
            # * step_size = (ub - lb) * 1/number_of_steps
            (boundary[1] - boundary[0]) * (1 / (self.number_of_steps[i] - 1))
            for i, boundary in enumerate(self._boundaries))

    def rasterize_get_cell(self, raster_num):
        """
        Create a new Cell.

        Returns a new Cell for a specif number in the raster.

        Parameters
        ----------
        raster_num : int
            Step number

        Returns
        -------
        cell : Cell

        """
        return self.optimizer_interface(self.rasterize_get_parameters(raster_num))

    def optimizer_boundaries(self):
        """
        Like get_transformations_boundaries(), except it returns a 1-dim list.

        """
        return [item for sublist in self.get_transformations_boundaries()
                for item in sublist]

    def optimizer_interface(self, args):
        """
        Calls all transformations in the order of their addition.

        All transformations are performed with the arguments as a 1-dim
        list, which is handy for an optimizer.

        Parameters
        ----------
        args : array
            1-dimensional list of all arguments.

        Returns
        -------
        cell : Cell()
            A cell on which all transformations have been performed.

        """
        if len(args) == len([item for sublist in
                             self.get_transformations_boundaries() for item in
                             sublist]):
            formatted_arguments = []
            i = 0
            for transformation in self._transformations:
                formatted_arguments.append(args[i:i + len(transformation[2])])
                i += len(transformation[2])
            return self.do_transformations(
                list(range(len(self._transformations))), formatted_arguments)
        else:
            raise ValueError("Number of arguments doesn't match number of" +
                             'dynamic arguments.')
