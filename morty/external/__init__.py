"""
external
--------

This package bundles external programs and tools that are used by *senkerlib*.

JSMOL
=====
The `JSMOL <http://wiki.jmol.org/index.php/Jmol_JavaScript_Object>`_
version used by :class:`senkerlib.util.jmol`. The folder has been extracted
from the standard Jmol distribution.
The following should be kept in mind when updating:

    * We got rid of the complete *java/* and data folder as well as *\*htm* and
      *\*html* files in the top and the *jsme* folder to keep the size of the
      repository as small as possible.
    * The files aciii_logo_orig.png, aciii_logo.png and JmolPopIn.js have
      been added.



Literature 
========== 
The folder literature holds (at the moment one) bibtex file providing the 
citations used in docstrings throughout senkerlib.

"""
