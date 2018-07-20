"""
s


"""

from .tensor import DipoleTensor


__all__ = ['Spinsystem']


class Spinsystem:
    """
    Spinsystem class that can be used by :class:`morty.calculate.SimpsonCaller`
    to perform calculations on spinsystems dynamically created from structure models.

    """

    def __init__(self, nuclei, dipoles=None, my_atom_numbers=None):
        """
        Creates a new Spinsystem instance.

        Parameters
        ----------
        nuclei : array_like
            List of :class:`morty.atomistic.Atom` instances.
        dipoles : list of tuples [(n1, n2, sc_f), ...]
            List of tuples handing over the indices of the two nuclei for
            featuring the dipole coupling, as well as the scaling factor to
            be applied to the resulting dipole coupling constant.
        my_atom_numbers : list
            The atomnumbers that make up the spinsystem.

        """
        if dipoles is None:
            dipoles = []
        self.nuclei = nuclei
        self.dd_couplings = []
        self.my_atom_numbers = my_atom_numbers

        for coupling in dipoles:
            self.dd_couplings.append(
                (coupling[0], coupling[1],
                 DipoleTensor(atom1=self.nuclei[coupling[0]],
                              atom2=self.nuclei[coupling[1]]),
                 coupling[2]))

    @staticmethod
    def all_couplings(num):
        """
        Returns a list with all possible connections between nuclei.

        Can be used for the dipoles argument of the constructor. You would use
        it like `dipoles=Spinsystem.all_couplings(len(allatoms))`.

        """
        connections = []
        for i in range(0, num):
            for j in range(i + 1, num):
                connections.append((i, j))
        return connections
