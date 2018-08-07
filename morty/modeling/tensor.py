"""
Various types of tensors used to describe NMR parameters.


"""
import math
import numpy as np
import scipy.constants

from ..util import zyz_euler_matrix
from .. import constants
from ..analytical import ddevolution


__all__ = ['CSATensor', 'DipoleTensor', 'EFGTensor',
           'discrete_average_csa_tensor']


class Tensor:
    """
    General tensor class.

    Class to handle stuff like csa, dipole and quadrupolar tensors, including
    setting them up in the right (that is to say, SIMPSON style) conventions.
    Offers convenient methods for setting the tensors and retrieving the
    info again.
    """

    def __init__(self, tensor):
        """
        General tensor class.

        Parameters
        ----------
        tensor : np.ndarray
            If given, the tensor is set with the matrix representation.

        """

        self.tensor = tensor
        self.eigenbasis = None

    @staticmethod
    def _calculate_euler(eigenbasis):
        """
        Calculates Euler angles for ZYZ rotation from a given eigenbasis.

        See http://en.wikipedia.org/wiki/Euler_angles for the calculation
        of the ZYZ angles from a matrix.

        """
        # beta = 0, 180 makes sin(beta) = 0, so we have to calculate
        # differently. We can choose alpha or beta arbitrarily, so we set
        # gamma = 0.
        if 1 - 1e-11 < abs(eigenbasis[2, 2]) < 1 + 1e-11:
            return (math.atan2(eigenbasis[1, 0], eigenbasis[0, 0]),
                    math.acos(math.copysign(1, eigenbasis[2, 2])), 0)
            # used copysign here, because of roundoff error in eigenbasis[2, 2]
        # beta is ambiguous. We use the convention beta > 0, therefore we
        # take its abs() value and can calculate alpha and beta as follows.
        return (math.atan2(eigenbasis[1, 2], eigenbasis[0, 2]),
                abs(math.acos(eigenbasis[2, 2])),
                math.atan2(eigenbasis[2, 1], -eigenbasis[2, 0]))

    @staticmethod
    def _convert_basis_right_handed(basis):
        """
        Makes a set of basis vectors right-handed.

        This switches the sign of the y-vector, if the coordinate system is
        oriented left-handed.

        """
        # right handed means: if you look at the xy-plane from above and x and
        # y are positively oriented (counter-clockwise), the z axis points up!
        if float(np.cross(basis[:, 0].T, basis[:, 1].T).dot(basis[:, 2])) < 0:
            basis[:, 1] *= -1
        return basis

    @staticmethod
    def transform_pas_to_cas(tensor, euler):
        """
        Transforms tensor from PAS to CAS.

        Given a tensor in PAS and Euler angles, this will rotate the tensor
        into the CAS.

        Parameters
        ----------
        tensor : array_like
            3x3 tensor in PAS.
        euler : array_like
            Three Euler angles.

        Returns
        -------
        tensor_cas : np.array
            The rotated tensor.

        """
        # Active rotation, therefore mat.T * tensor * mat
        rot_mat = zyz_euler_matrix(euler[0], euler[1], euler[2])
        return rot_mat.T @ tensor @ rot_mat


class CSATensor(Tensor):
    """
    Handles CSA Tensors.

    Can calculate Euler angles, the values of isotropic shift, anisotropy and
    asymmetry according to Haeberlen-Mehring-Spiess.

    Notes
    -----
    Usually we will deal with shift tensors, that means the input from
    Gaussian/CASTEP is multiplied by -1 in Cell.load_*_tensor.
    BE AWARE that we do explicitly NOT use chemical shiedling here,
    but the chemical shift!

    """
    def __init__(self, tensor=None, hms=None,
                 euler_pas_to_cas=(0, 0, 0)):
        """
        Parameters
        ----------
        tensor : np.ndarray
            If given, the tensor is set with the matrix representation.

        Attributes
        ----------
        hms : tuple
            Tuple of (isotropic shift, anisotropy, asymmetry).
        eigenbasis : np.ndarray
            Eigenbasis of the tensor.
        euler_pas_to_cas : tuple
            Euler angles (alpha, beta, gamma) for transform from PAS to CAS.
        tensor : np.ndarray
            Tensor in CAS.
        tensor_pas : np.ndarray
            Tensor in PAS.

        """
        super().__init__(tensor)
        # (isotropic shift, anisotropy, asymmetry
        self.hms = (None, None, None)
        self.tensor_pas = None

        if tensor is not None:
            self.set_cas_tensor(tensor)
        elif hms is not None:
            self.set_hms(hms, euler_pas_to_cas)
        else:
            raise RuntimeError('You need to supply either a tensor matrix or Haberlen-Mehring-Spiess values.')

    @staticmethod
    def _calculate_hms(tensor):
        """
        Calculates Haberlen-Mehring-Spiess parameters from a tensor in PAS.

        Parameters
        ----------
        tensor: tensor()
            A tensor, expressed within the PAS.

        Returns
        -------
        hms: list()
            List containing isotropic shift, anisotropy and asymmetry.

        """
        iso = 1 / 3 * np.trace(tensor)
        aniso = tensor[2, 2] - iso
        eta = (tensor[1, 1] - tensor[0, 0]) / aniso
        return (iso, np.float(aniso), np.float(eta))

    def set_reference(self, difference):
        """
        Corrects the isotropic shift of the tensor.

        This function adds a given value to the isotropic shift of the tensor.

        Parameters
        ----------
        difference : float
            The value that is added to the isotropic shift (usually in ppm).

        """
        self.set_hms((self.hms[0] + difference, self.hms[1], self.hms[2]),
                     self.euler_pas_to_cas)

    def set_cas_tensor(self, tensor, aniso_sign=None):
        """
        Sets the tensor with a matrix representation.

        Calculates all tensor parameters from a given matrix. If the matrix
        is in the CAS, Euler angles are automatically calculated.

        Parameters
        ----------
        tensor : np.ndarray
            Matrix representation of the tensor in CAS or PAS.

        """
        # we completely forget about the real tensor and just store the
        # symmmetrized tensor (see Levitt, Spin Dynamics: the CSA tensor might
        # be not symmetric, but you can only observe the symmetrized tensor)
        self.tensor = .5 * (tensor + tensor.T)

        # find eigenvalues and vectors
        eigenvals, eigenvecs = np.linalg.eigh(self.tensor)

        self.set_eigen(eigenvals, eigenvecs, aniso_sign)

    def set_eigen(self, eigenvalues, eigenvectors, aniso_sign=None):
        """
        Set the tensor with its eigenvalues and eigenvectors.

        Parameters
        ----------
        eigenvalues : array
            Array of all eigenvalues.
        eigenvectors : np.ndarray 
            Matrix with three eigenvectors (they don't need to be in the right
            order).
        aniso_sign : -1/1
            For eta = 1, the sign of the anisotropy is ambiguous.

        """
        # sort the eigenvalues/vectors according to Haeberlen-Mehring-Spiess:
        # |s_zz - s_iso| >= |s_xx - s_iso| >= |s_yy - s_iso|
        order = list(np.abs(eigenvalues - 1 / 3 * np.sum(eigenvalues)))
        order_sorted = sorted(order)
        sigmazz = eigenvalues[order.index(order_sorted[2])]
        vecz = np.array(eigenvectors)[:, order.index(order_sorted[2])]
        sigmayy = eigenvalues[order.index(order_sorted[0])]
        vecy = np.array(eigenvectors)[:, order.index(order_sorted[0])]
        sigmaxx = eigenvalues[order.index(sorted(order)[1])]
        vecx = np.array(eigenvectors)[:, order.index(order_sorted[1])]
        # Save the eigenbasis as a row matrix. Do this to get the Euler angles
        # from PAS to CAS for an active rotation, (i.e. CAS = R^-1 * PAS * R)
        # Also, switch sign of z axis if necessary
        self.eigenbasis = self._convert_basis_right_handed(
            np.array([vecx, vecy, vecz]).T).T

        # If eta = 1, aniso is ambiguous.
        if aniso_sign is not None:
            if np.sign(aniso_sign) != np.sign(sigmazz):
                sigmaxx, sigmazz = sigmazz, sigmaxx
                vecx, vecz = vecz, vecx

        self.euler_pas_to_cas = self._calculate_euler(self.eigenbasis)
        self.tensor_pas = np.diag((sigmaxx, sigmayy, sigmazz))

        self.hms = self._calculate_hms(self.tensor_pas)

    def set_hms(self, hms, euler_pas_to_cas=None):
        """
        Set the tensor properties in Haeberlen-Mehring-Spiess convention.

        Parameters
        ----------
        hms : tuple of floats (isotropic, anisotropy, asymmetry)
            Tuple of isotropic chemical shift, anisotropy and asymmetry.
        euler_pas_to_cas : tuple of floats (alpha, beta, gamma)
            Euler angles in radians for transformation from PAS to CAS.

        """

        self.hms = hms
        self.tensor_pas = np.diag((hms[0] - .5 * hms[1] * (1 + hms[2]),
                                   hms[0] - .5 * hms[1] * (1 - hms[2]),
                                   hms[0] + hms[1]))

        if euler_pas_to_cas is not None:
            aniso_sign = None
            if hms[2] == 1.0:
                aniso_sign = np.sign(hms[1])
            self.set_cas_tensor(self.transform_pas_to_cas(self.tensor_pas,
                                                          euler_pas_to_cas),
                                aniso_sign=aniso_sign)

    def get_anisotropic_tensor(self):
        """
        Returns the anisotropic part of the tensor.

        Returns
        -------
        tensor : np.ndarray

        """
        return self.tensor - np.diag((self.hms[0],) * 3)


class DipoleTensor(Tensor):
    """
    Handles DD interaction.

    Attributes
    ----------
    distance : float
        The distance of the contributing atoms in Å.
    eigenbasis : np.ndarray
        Unitless tensor representing the orientation of the dd coupling tensor.
    direct_dipolar_coupling_tensor : np.ndarray
        Dipolar coupling tensor.
    euler_pas_to_cas : np.ndarray
        The euler angles of the dd couplign tensor.
    coupling_constant : float
        The dipolar coupling constant, not divided by 2π as SIMPSON does.

    """
    def __init__(self, atom1=None, atom2=None, my_tensor=None,
                 my_isotope1=None, my_isotope2=None):
        r"""
        Set the dipolar coupling tensor.

        Set the dipolar coupling tensor either by supplying the tensor itself
        or the contributing atoms.

        Parameters
        ----------
        atom1, atom2 : :class:`morty.modeling.Atom`, optional
            The contributing atoms. Naturally, if you want to define the
            tensor using these atoms, you have to hand over both atoms. The
            atoms are preferred over *mytensor*.
        my_tensor : np.ndarray, optional
            Set the tensor directly. *my_isotope\** is mandatory if *my_tensor*
            is set.
        my_isotope1, my_isotope2 : str, optional
            Sets the isotopes for the two atoms, e.g. '1H' or '13C'.

        Notes
        -----
        The dipolar coupling tensor in the PAS has the form

        .. math::
           \mathbf{D_{PAS}} =
            \left(\begin{matrix}
            -d/2 & 0 & 0\\
            0 & -d/2 & 0 \\
            0 & 0 & d
            \end{matrix}\right)

        The eigenbasis takes the form

        .. math::
           \mathbf{E_{PAS \rightarrow CAS}} =
            \left(\begin{matrix}
            a1   & b1   & c1 \\
            a2   & b2   & c2 \\
            a3   & b3   & c3
            \end{matrix}\right)

        The :math:`\vec{c}` vector aligns with the axis between the two atoms
        involved, the :math:`\vec{a}` vector is perpendicular to
        :math:`\vec{c}` and chosen to be parallel to the x/y plane of the
        **CAS**, and finally the :math:`\vec{b}` is perpendicular to both
        :math:`\vec{a}` and :math:`\vec{c}`.

        To get the direct dipolar coupling tensor, one multiplies the columns
        of the eigenbasis with the respective length known from the **PAS**.

        The dipolar coupling constant is defined as:

        .. math::
            d_{IS} = \frac{\gamma_I \gamma_S \hbar}{r^3_{IS}}
                \frac{\mu_0}{4\pi}

        """
        self._atom1 = None
        self._atom2 = None
        self.coupling_constant = None
        self.direct_dipolar_coupling_tensor = None
        self.distance = None
        self.eigenbasis = None
        self.euler_pas_to_cas = None
        if atom1 is not None and atom2 is not None:
            self.set_tensor_by_atoms(atom1, atom2)
        elif my_tensor is not None and my_isotope1 and my_isotope2:
            self.set_tensor_cas(my_tensor, my_isotope1,
                                my_isotope2)
        else:
            raise Warning('Creating empty tensor.')

    def set_tensor_by_atoms(self, atom1=None, atom2=None):
        """
        Determine coupling tensor between two atoms.

        Parameters
        ----------
        atom1, atom2 : :class:`morty.modeling.Atom`
            Atoms from which the coupling tensor will be calculated.

        """
        self._atom1 = atom1
        self._atom2 = atom2

        self.distance = np.linalg.norm(atom1.position_abs - atom2.position_abs)

        self.coupling_constant = ddevolution.dd_coupling_constant(
            self.distance, atom1.get_gyromagnetic_ratio(),
            atom2.get_gyromagnetic_ratio())
        zvec = atom2.position_abs - atom1.position_abs
        yvec = np.cross([0, 0, 1], zvec)
        xvec = np.cross(zvec, yvec)
        self.eigenbasis = np.array([xvec / np.linalg.norm(xvec),
                                    yvec / np.linalg.norm(yvec),
                                    zvec / np.linalg.norm(zvec)])
        self.euler_pas_to_cas = self._calculate_euler(self.eigenbasis)

        self.tensor_pas = np.diag((-self.coupling_constant, -self.coupling_constant,
                                   self.coupling_constant * 2))
        self.tensor_cas = self.transform_pas_to_cas(self.tensor_pas, self.euler_pas_to_cas)

    def set_tensor_cas(self, my_tensor=None, my_isotope1=None,
                       my_isotope2=None):
        """
        Set the dipolar coupling tensor directly.

        This function sets the dipolar coupling tensor directly, meaning you
        hand over the dipolar coupling tensor (which in the PAS is
        [[-d, 0, 0],[0, -d, 0],[0, 0, 2*d]]) and not the eigenbasis.

        Parameters
        ----------
        my_tensor : np.ndarray
            The direct dipolar coupling tensor.
        my_isotope1, my_isotope2 : str
            The isotopes to use for the coupling.

        """
        self.tensor_cas = .5 * (my_tensor + my_tensor.T)

        # find eigenvalues and vectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.tensor_cas)

        order = list(np.abs(eigenvalues - 1 / 3 * np.sum(eigenvalues)))
        order_sorted = sorted(order)
        sigmazz = eigenvalues[order.index(order_sorted[2])]
        vecz = np.array(eigenvectors)[:, order.index(order_sorted[2])]
        sigmayy = eigenvalues[order.index(order_sorted[0])]
        vecy = np.array(eigenvectors)[:, order.index(order_sorted[0])]
        sigmaxx = eigenvalues[order.index(sorted(order)[1])]
        vecx = np.array(eigenvectors)[:, order.index(order_sorted[1])]
        # Save the eigenbasis as a row matrix. Do this to get the Euler angles
        # from PAS to CAS for an active rotation, (i.e. CAS = R^-1 * PAS * R)
        # Also, switch sign of z axis if necessary
        self.eigenbasis = np.array([vecx, vecy, vecz])
        #self.eigenbasis = self._convert_basis_right_handed(
        #    np.array([vecx, vecy, vecz]).T).T

        self.euler_pas_to_cas = self._calculate_euler(self.eigenbasis)
        self.tensor_pas = np.diag((sigmaxx, sigmayy, sigmazz))
        sign_d = (
            math.copysign(1, constants.GYROMAGNETIC_RATIOS[my_isotope1]) *
            math.copysign(1, constants.GYROMAGNETIC_RATIOS[my_isotope2]))
        if math.copysign(1, self.tensor_pas[2, 2]) == sign_d:
            self.tensor_pas *= -1

        #self.coupling_constant = self.tensor_pas[2, 2] / 2
        self.coupling_constant = (self.tensor_pas[2, 2] - self.tensor_pas[0, 0]  -
                                  self.tensor_pas[1, 1]) / 4
        self.euler_pas_to_cas = self._calculate_euler(self.eigenbasis)
        self.distance = ddevolution.dist_from_dd_coupling_constant(
            self.coupling_constant,
            constants.GYROMAGNETIC_RATIOS[my_isotope1],
            constants.GYROMAGNETIC_RATIOS[my_isotope2])

        self.eta = (self.tensor_pas[1, 1] - self.tensor_pas[0, 0]) / self.tensor_pas[2, 2]


class EFGTensor(Tensor):
    r"""
    Handles quadrupolar Tensors.

    Can calculate Euler angles as well as the values of cq and asymmetry.


    Attributes
    ----------
    atom : :class:`morty.modeling.Atom`
        Atom() instance for which CQ is calculated. If this is defined (as it
        is done by reading a magres file), CQ will be calculated with the
        isotope of this atom, every time it is read. If this is not set, the
        attribute `isotope` is used.
    isotope : str
        Key thats present in constants.quadrupole_moment. E.g. '2H' or
        '27Al'. If atom is set, this is not used.
    cq_eta : tuple
        Tuple of (cq, eta).
    eigenbasis : np.ndarray
        Eigenbasis of the tensor.
    euler_pas_to_cas : tuple
        Euler angles (alpha, beta, gamma) for transform from PAS to CAS.
    tensor : np.ndarray
        Tensor in CAS.
    tensor_pas : np.ndarray
        Tensor in PAS.

    Notes
    -----
    There are, as always, several conventions of how to define Euler angles,
    or how to sort the eigenvalues. Here, we will use the same definition
    of Euler angles as for the CSA (which is the one adopted by SIMPSON).
    We also are using the SIMPSON convention for ordering the components of
    the magnitudes of the EFG tensor V:

    .. math::
        V_{zz} \ge V_{xx} \ge V_{yy}

    The asymmetry parameter is therefore defined as:

    .. math::
        \eta = \frac{V_{yy} - V_{xx}}{V_{zz}}

    Please be aware that according to Klaus Eichele, most people, including
    the NQR community (See Frank Haarmann contribution to eMagRes
    :cite:`Haarmann2011`), use another convention, where :math:`V_{yy} \ge
    V_{xx}`.

    The quadrupolar coupling constant is defined as:

    .. math::
        C_{Q} = \frac{e \cdot V_{zz} \cdot Q}{h}

    As always, units are a bitch. CASTEP calculates the electric field
    to :math:`[V] = \mathrm{\frac{eV}{Å \cdot e}}`,
    and therefore the electric field gradient
    to :math:`[V_{zz}] = \mathrm{\frac{eV}{Å^2 \cdot e}}`
    As we are using SI units, we have to add conversion factors to
    get :math:`[V_{zz}] = \mathrm{\frac{J}{C \cdot m^2}}`.
    The magnitude of e will cancel out
    when converting from casteps "natural" to our SI unit system via
    :math:`[V_{zz}] = \mathrm{\frac{e}{e \cdot 10^{-20}}}`.

    """
    def __init__(self, tensor=None, atom=None, isotope=None):
        """
        Handles quadrupolar Tensors.

        Parameters
        ----------
        tensor : np.ndarray
            If given, the tensor is set with the matrix representation.
        atom: :class:`morty.modeling.Atom`
            Reference to the atom to use. CQ will be calculated according to
            its isotope.
        isotope : str
            If atom is not set, this will be used to calculate CQ. Must be
            included in constants.quadrupole_moment, e.g. '2H'.

        """
        super().__init__(tensor)

        self.atom = atom
        self._isotope = isotope
        if tensor is not None:
            self.set_tensor(tensor)
        else:
            self.tensor_pas = None
            self.euler_pas_to_cas = (None, None, None)

    @property
    def isotope(self):
        """
        Holds the isotope of the atom.

        """
        if self.atom is not None:
            return str(self.atom.mass) + str(self.atom.atom_type)
        return self._isotope

    @property
    def cq(self):
        """
        Returns the coupling constant CQ, if the isotope is known.

        Returns
        -------
        cq : float
            CQ in Hz.

        """
        # unit of gradient of electric field strength =
        # 9.71736469e21 Vm^-2
        try:
            return self.tensor_pas[2, 2] * (scipy.constants.e * 9.71736e21 / scipy.constants.h *
                                            constants.QUADRUPOLE_MOMENT[self.isotope])
        except KeyError:
            return 0

    @property
    def eta(self):
        """
        Returns the asymmetry parameter for the EFG tensor.

        Returns
        -------
        eta : float

        """
        return (self.tensor_pas[1, 1] - self.tensor_pas[0, 0]
               ) / self.tensor_pas[2, 2]

    def set_tensor(self, tensor):
        """
        Sets the tensor with a matrix representation.

        Calculates all tensor parameters from a given matrix. If the matrix
        is in the CAS, Euler angles are automatically calculated.

        Parameters
        ----------
        tensor : np.ndarray
            Matrix representation of the tensor in CAS or PAS.

        """

        self.tensor = tensor
        # find eigenvalues and vectors
        eigenvals, eigenvecs = np.linalg.eigh(self.tensor)

        self.set_eigen(eigenvals, eigenvecs)

    def set_eigen(self, eigenvalues, eigenvectors):
        """
        Set the tensor with its eigenvalues and eigenvectors.

        Parameters
        ----------
        eigenvalues : array
            Array of all eigenvalues.
        eigenvectors : np.ndarray
            Matrix with three eigenvectors (they don't need to be in the right
            order).

        """
        # sort the eigenvalues/vectors according to Haeberlen-Mehring-Spiess:
        # |s_zz - s_iso| >= |s_yy - s_iso| >= |s_xx - s_iso|
        order = list(np.abs(eigenvalues - 1 / 3 * np.sum(eigenvalues)))
        order_sorted = sorted(order)
        v_zz = eigenvalues[order.index(order_sorted[2])]
        vecz = np.array(eigenvectors)[:, order.index(order_sorted[2])]
        v_yy = eigenvalues[order.index(order_sorted[0])]
        vecy = np.array(eigenvectors)[:, order.index(order_sorted[0])]
        v_xx = eigenvalues[order.index(sorted(order)[1])]
        vecx = np.array(eigenvectors)[:, order.index(order_sorted[1])]
        # Save the eigenbasis as a row matrix. Do this to get the Euler angles
        # from PAS to CAS for an active rotation, (i.e. CAS = R^-1 * PAS * R)
        # Also, switch sign of z axis if necessary
        self.eigenbasis = self._convert_basis_right_handed(
            np.array([vecx, vecy, vecz]).T).T

        self.euler_pas_to_cas = self._calculate_euler(self.eigenbasis)
        self.tensor_pas = np.diag((v_xx, v_yy, v_zz))

    def set_cq_eta(self, cq_eta, euler_pas_to_cas=None):
        """
        Set the tensor properties via cq and eta.

        Parameters
        ----------
        ca_eta : tuple of floats (cq, asymmetry)
            Tuple of cq and asymmetry.
        euler_pas_to_cas : tuple of floats (alpha, beta, gamma)
            Euler angles in radians for transformation from PAS to CAS.

        """
        myvzz = cq_eta[0] / (scipy.constants.e * 9.71736e21 / scipy.constants.h *
                             constants.QUADRUPOLE_MOMENT[self.isotope])
        self.tensor_pas = np.array([[- .5 * myvzz * (1 + cq_eta[1]), 0.0,
                                     0.0],
                                    [0.0, - .5 * myvzz * (1 - cq_eta[1]),
                                     0.0],
                                    [0.0, 0.0, myvzz]])

        if euler_pas_to_cas != None:
            self.set_tensor(self.transform_pas_to_cas(
                self.tensor_pas, euler_pas_to_cas))
        else:
            self.tensor = self.tensor_pas


def discrete_average_csa_tensor(tensors):
    """
    Averages CSA tensors for discrete jumps (fast motion limit).

    Parameters
    ----------
    tensors : array of (tensor : CSATensor(), weighting : float)
        The tensors and weightings for averaging. The sum of all weightings
        must be 1.

    Returns
    -------
    averageTensor : CSATensor()

    """
    return CSATensor(np.sum((myTensor[0].tensor * myTensor[1]
                             for myTensor in tensors)))
