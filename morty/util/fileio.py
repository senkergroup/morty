"""
Handling of input/output from and to files.
"""

import os
import numpy as np


def find_type_of_string(mystring):
    """
    Return the string as the type it is.

    The string is checked for its type and is returned as the type it is.
    Types that are recognised:

    - int
    - float
    - bool
    - str

    """
    try:
        mystring = int(mystring)
    except ValueError:
        try:
            mystring = float(mystring)
        except ValueError:
            if mystring == 'True':
                mystring = True
            elif mystring == 'False':
                mystring = False
            else:
                mystring = mystring
    return mystring


def determine_filename(target, fileending, filename=None, foldername=None):
    """
    Determine a suitable filename for files to read in to a cell.

    If no filename and/or foldername are given, the respective *cellname* and
    *foldername* parameters of the target cell are used to construct sensible
     defaults.

    Parameters
    ----------
    target : :class:`morty.atomistic.Cell`
        The target cell.
    fileending : str
        The target file ending, e.g. '.cell'.
    filename : str
        The filename of the file. May also be a absolute path. In that case,
        `foldername` is overridden.
    foldername : str
        The foldername to use. `+` concatenates with *target.foldername*,
        otherwise overrides the latter.

    Returns
    -------
    myfilename : str
        The absolute path of the file.

    """
    if foldername is None:
        foldername = os.path.abspath(target.foldername)
    elif foldername[0] == '+':
        foldername = os.path.abspath(os.path.join(target.foldername,
                                                  foldername[1:]))
    else:
        foldername = os.path.abspath(foldername)

    if not filename:
        if target.cellname:
            myfilename = os.path.abspath(os.path.join(foldername,
                                                      target.cellname +
                                                      fileending))
        else:
            myfilename = os.path.abspath(search_file_ending_with(foldername,
                                                                 fileending))
    else:
        if foldername != './':
            myfilename = os.path.abspath(os.path.join(foldername,
                                                      filename))
        else:
            myfilename = os.path.abspath(filename)
    return myfilename


def search_file_ending_with(directory, ext):
    """
    Searches for a file with the specified file extension within the specified
    directory.

    Parameters
    ----------
    directory : string
        The directory to search within.
    ext : string
        The file extension for which to return the filename.

    Returns
    -------
    myfile : string
        The path of the file, including the filename, relative to
        `directory`.
    """
    files = os.listdir(directory)
    check = None
    for myfile in files:
        if myfile.endswith(ext):
            check = myfile
    if check is None:
        return None
    return os.path.join(directory, check)


def search_files_ending_with(directory, ext):
    """
    Searches for all files with the specified file extension within the
    specified directory.

    Parameters
    ----------
    directory : string
        The directory to search within.
    ext : string
        The file extension for which to return the filename.

    Returns
    -------
    myfile : list
        A list containing the path, including the filename, for each file,
        relative to `directory`.
    """
    files = os.listdir(directory)
    thefilesfound = []
    for myfile in files:
        if myfile.endswith(ext):
            if directory == '.' or directory == './':
                thefilesfound.append(myfile)
            else:
                thefilesfound.append(directory + '/' + myfile)
    return thefilesfound


class SsParser():
    """
    Class to parse the StructureSol configuration file format.

    To keep backwards compatibility with our older programs, this
    reads our special config file format.

    Blocks are defined encapsulated by: ::

        %BLOCKNAME
        ...
        %END_BLOCKNAME

    Within blocks there are simple key/value pairs, delimited by whitespaces,
    ':' or '='.

    """
    def __init__(self, filename=None):
        """
        Instantiates a SsParser.

        Parameters
        ----------
        filename : str, optional
            If given, a file is read.

        """
        self.blocks = {}
        if filename is not None:
            self.read(filename)

    @staticmethod
    def _parse_line(line):
        delimiters = [' ', ':', '=']

        position = -1
        found_delimiter = ''
        for delimiter in delimiters:
            current_position = line.find(delimiter)
            if ((current_position < position or position == -1) and
                    current_position != -1):
                position = current_position
                found_delimiter = delimiter

        split = line.split(found_delimiter)
        value = ''
        for part in split[1:]:
            value += part + found_delimiter
        # maybe we just found a whitespace yet... might be followed by another
        # delimiter! (which might be followed by another whitespace!)
        for delimiter in delimiters:
            value = value.strip(delimiter)
        value = value.strip()

        return split[0], value

    def _append_block(self, block, keyvalues):
        block = block.lower()
        if not isinstance(self.blocks.get(block), list):
            self.blocks[block] = []

        self.blocks[block].append(keyvalues)

    def read(self, filename):
        """
        Reads in a file.

        Parameters
        ----------
        filename : str
            The file to read.

        """
        file = open(filename, 'r')

        line = file.readline()
        while line != '':
            line = line.strip()
            if line.startswith('%'):
                my_block = line[1:].lower()
                my_key_values = {}
                line = file.readline()
                while line.lower() != '%end_' + my_block:
                    line = line.strip()
                    if not line.startswith('#') and line != '':
                        key, value = self._parse_line(line)
                        my_key_values[key] = find_type_of_string(value)
                    line = file.readline()
                    line = line.strip()
                self._append_block(my_block, my_key_values)

            line = file.readline()
        file.close()

    @staticmethod
    def _getboolean(string):
        if (string.lower() == 'no' or string.lower() == 'false' or
                string == '0' or string == ''):
            return False
        return True

def read_in_magres_nics(magres_nics_file):
    """
    Reads in a CASTEP ``magres_nics`` file.

    Parameters
    ----------
    magres_nicsfile : string, optional
        The path to the ``magres_nics`` file to read in. May be relative.
        Defaults to './'.

    Returns
    -------
    nics : array
        Returns the content of the ``magres_nics`` file as a numpy array.
    """
    file = open(magres_nics_file, 'r')
    nics = []
    _nics = []
    linecounter = 0
    for line in file:
        line = line.strip()
        if line != '':
            _nics.append([float(x) for x in line.split()])

            if linecounter == 4:
                linecounter = -1
                nics.append(_nics)
                _nics = []
            linecounter += 1

    return np.array(nics)
