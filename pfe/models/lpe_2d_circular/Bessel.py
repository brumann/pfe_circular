"""Duct mode boundary condition for the LPE in 2D"""
from numpy.lib.scimath import sqrt
from scipy.spatial.distance import pdist, squareform
from math import pi
import numpy as np
import scipy.integrate as it
import scipy.optimize as opt
import scipy.special as sp
from .utils import *

class Bessel:
    # Bessel class dedicated to the definition of Bessel functions and matrices.

    def __init__(self, type=None, x=None, r1=None, r2=None, inv_Z1=None, inv_Z2=None,
                 D0=None, U0=None, C0=None, w=None, m=None, nb_n=None, scattering='active'):
        # PUBLIC ATTRIBUTES
        #: (*str*) -- Type of geometry and flow ('cst' or 'evol_x')
        self.type = type
        #: (*ndarray*) -- Axial coordinate
        self.x = x
        #: (*ndarray*) -- Hub radius
        self.r1 = r1
        #: (*ndarray*) -- Tip radius
        self.r2 = r2

        #: (*ndarray*) -- Inverse of hub impedance
        self.inv_Z1 = inv_Z1
        #: (*ndarray*) -- Inverse of tip impedance
        self.inv_Z2 = inv_Z2

        #: (*ndarray*) -- Mean density
        self.D0 = D0
        #: (*ndarray*) -- Mean speed of sound
        self.C0 = C0
        #: (*ndarray*) -- Mean axial velocity
        self.U0 = U0

        #: (*float*) -- Pulsation
        self.w = w
        #: (*int*) -- Azimuthal mode order
        self.m = m
        #: (*int*) -- Number of radial modes to be used
        self.nb_n = nb_n

        #: (*str*) -- Scattering to be chosen between ['active', 'inactive'] -- Default value: 'active'
        self.scattering = scattering

        # PROTECTED ATTRIBUTES
        #: (*int*) -- Number of Legendre points to be used in the radial direction
        self._nb_leg = 500

        #: (*ndarray*) -- Radial eigenvalues evolution (nb_x, nb_n)
        self._alpha = None

        #: (*dict*) -- dictionnary containing the Chebyshev matrices: int_r1^r2 Tn Tm r dr
        self._matrices = {}

    def compute_radial_eigvals_evolution(self, inv_Z1=None, inv_Z2=None, first_eigvals=None):
        """
        Compute the radial eigenvalues for each axial position.

        :param complex inv_Z1: if not None, allows to force a value different form self.inv_Z1 for the inverse of hub impedance
        :param complex inv_Z2: if not None, allows to force a value different form self.inv_Z2 for the inverse of tip impedance
        :param array_like first_eigvals: duct eigenvalues at the first axial location (None if not known)

        :returns: array of shape (nb_x, nb_n) containing the first nb_n radial eigenvalues at each axial position
        :rtype: ndarray

        """
        # get attributes
        x = self.x
        if inv_Z1 is None:
            inv_Z1 = self.inv_Z1
        if inv_Z2 is None:
            inv_Z2 = self.inv_Z2
        nb_n = self.nb_n

        # initialize results arrays
        nb_x = len(x)
        all_alpha = np.zeros((nb_x, nb_n), dtype=complex)
        if first_eigvals is None:
            first_eigvals = self.compute_sorted_radial_eigvals(0, return_all=False)[0]
        all_alpha[0, :] = first_eigvals

        # compute eigenvalues for each axial position
        for id_x in range(1, nb_x):
            if (inv_Z1[id_x] == 0.) and (inv_Z2[id_x] == 0.):
                alpha = self.compute_radial_eigvals_hard(id_x, last_eigvals=all_alpha[id_x - 1, :])
            else:
                alpha = self.compute_radial_eigvals_lined(id_x, 1., last_eigvals=all_alpha[id_x - 1, :])

            # follow modes
            id_order = np.abs(np.subtract.outer(alpha, all_alpha[id_x - 1, :])).argmin(0)
            all_alpha[id_x, :] = alpha[id_order]

        self._alpha = all_alpha
        # return results
        return all_alpha

    def compute_sorted_eigvals(self, id_x=0, inv_Z1=None, inv_Z2=None, norm='surf', return_all=False):
        """
        Compute the sorted eigenvalues for hardwalled or lined duct with circular or annular cross section.

        :param int id_x: index that indicates the axial position at which the sorted eigenvalues are computed
        :param complex inv_Z1: if not None, allows to force a value different form inv_Z1[id_x] for the inverse of hub impedance
        :param complex inv_Z2: if not None, allows to force a value different form inv_Z2[id_x] for the inverse of tip impedance
        :param str norm: kind of normalization ('surf', 'tip')
        :param bool return_all: if True, return eigenmodes and evolution of eigenvalues from the hardwalled case

        :returns: tuple containing:

            - a 1d numpy array containing the first nb_n eigenvalues associated with right-running modes (*ndarray*)
            - a 1d numpy array containing the first nb_n eigenvalues associated with left-running modes (*ndarray*)
        :rtype: tuple of two elements

        """
        # get attributes
        if self.type is 'cst':
            r2 = self.r2
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2
            U0 = self.U0
            C0 = self.C0
        elif self.type is 'evol_x':
            r2 = self.r2[id_x]
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1[id_x]
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2[id_x]
            U0 = self.U0[id_x]
            C0 = self.C0[id_x]
        w = self.w
        nb_n = self.nb_n

        # compute radial eigenvalues
        alpha_dn, alpha_up = self.compute_sorted_radial_eigvals(id_x, inv_Z1, inv_Z2, return_all)

        # compute square of reduced axial wavenumber (sigma2)
        sigma2_dn = 1. - (C0 ** 2 - U0 ** 2) * alpha_dn ** 2 / w ** 2
        sigma2_up = 1. - (C0 ** 2 - U0 ** 2) * alpha_up ** 2 / w ** 2

        # transform to complex number to allow the use of np.sqrt even if sigma2 is negative and real
        sigma2_dn = sigma2_dn.astype(complex)
        sigma2_up = sigma2_up.astype(complex)

        # compute reduced axial wavenumber (sigma)
        # convention for hardwalled case:
        # downstream propagation -> Re(sigma) > 0 if cut-on, Im(sigma) < 0 if cut-off
        # upstream propagation -> Re(sigma) < 0 if cut-on, Im(sigma) > 0 if cut-off
        sigma_dn = np.where(np.sqrt(sigma2_dn).imag >= 1e-8, -np.sqrt(sigma2_dn), np.sqrt(sigma2_dn))
        sigma_up = np.where(np.sqrt(sigma2_up).imag <= 1e-8, -np.sqrt(sigma2_up), np.sqrt(sigma2_up))

        # compute axial wavenumber (mu)
        mu_dn = w * (C0 * sigma_dn - U0) / (C0 ** 2 - U0 ** 2)
        mu_up = w * (C0 * sigma_up - U0) / (C0 ** 2 - U0 ** 2)

        # return results
        if return_all:
            # get eigenmodes
            phi_norm_dn = self.get_radial_eigmods(id_x, alpha_dn[-1, :], 1., inv_Z1, inv_Z2, norm)
            phi_norm_up = self.get_radial_eigmods(id_x, alpha_up[-1, :], -1., inv_Z1, inv_Z2, norm)
            u_norm_dn = lambda r: -1j * mu_dn[-1, :] * phi_norm_dn(r)
            u_norm_up = lambda r: -1j * mu_up[-1, :] * phi_norm_up(r)

            if norm is 'tip':
                phi_norm_surf_dn = self.get_radial_eigmods(id_x, alpha_dn[-1, :], 1., inv_Z1, inv_Z2, 'surf')
                phi_norm_surf_up = self.get_radial_eigmods(id_x, alpha_up[-1, :], -1., inv_Z1, inv_Z2, 'surf')
                norm_surf_tip_dn = phi_norm_surf_dn(r2) / phi_norm_dn(r2)
                norm_surf_tip_up = phi_norm_surf_up(r2) / phi_norm_up(r2)
            else:
                norm_surf_tip_dn = 1.
                norm_surf_tip_up = 1.

            return mu_dn, mu_up, phi_norm_dn, phi_norm_up, u_norm_dn, u_norm_up, np.eye(nb_n) / norm_surf_tip_dn, np.eye(nb_n) / norm_surf_tip_up
        else:
            return mu_dn, mu_up

    def compute_sorted_radial_eigvals(self, id_x=0, inv_Z1=None, inv_Z2=None, return_all=False):
        """
        Compute the sorted radial eigenvalues for hardwalled or lined duct with circular or annular cross section.

        :param int id_x: index that indicates the axial position at which the sorted radial eigenvalues are computed
        :param complex inv_Z1: if not None, allows to force a value different form inv_Z1[id_x] for the inverse of hub impedance
        :param complex inv_Z2: if not None, allows to force a value different form inv_Z2[id_x] for the inverse of tip impedance
        :param bool return_all: if True, return evolution of eigenvalues from the hardwalled case

        :returns: tuple containing:

            - a 1d numpy array containing the first nb_n radial eigenvalues associated with right-running modes (*ndarray*)
            - a 1d numpy array containing the first nb_n radial eigenvalues associated with left-running modes (*ndarray*)
        :rtype: tuple of two elements

        """
        # get attributes
        if self.type is 'cst':
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2
        elif self.type is 'evol_x':
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1[id_x]
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2[id_x]
        nb_n = self.nb_n

        # discretization of the impedance space
        nb_inv_Z = 1 if (inv_Z1 == 0. and inv_Z2 == 0) else 100
        varying_inv_Z1 = inv_Z1 * np.linspace(0., 1., nb_inv_Z)
        varying_inv_Z2 = inv_Z2 * np.linspace(0., 1., nb_inv_Z)

        # initialize results arrays
        all_alpha_dn = np.zeros((nb_inv_Z, nb_n), dtype=complex)
        all_alpha_up = np.zeros((nb_inv_Z, nb_n), dtype=complex)

        # first computation without impedance
        alpha_dn = self.compute_radial_eigvals_hard(id_x, last_eigvals=None)
        alpha_up = alpha_dn
        all_alpha_dn[0, :] = alpha_dn
        all_alpha_up[0, :] = alpha_up

        # computations with increasing inv_Z1 and inv_Z2
        for id_inv_Z in range(1, nb_inv_Z):
            alpha_dn = self.compute_radial_eigvals_lined(id_x, 1., inv_Z1=varying_inv_Z1[id_inv_Z], inv_Z2=varying_inv_Z2[id_inv_Z],
                                                         last_eigvals=all_alpha_dn[id_inv_Z - 1, :])
            alpha_up = self.compute_radial_eigvals_lined(id_x, -1., inv_Z1=varying_inv_Z1[id_inv_Z], inv_Z2=varying_inv_Z2[id_inv_Z],
                                                         last_eigvals=all_alpha_up[id_inv_Z - 1, :])

            # follow modes
            id_dn = np.abs(np.subtract.outer(alpha_dn, all_alpha_dn[id_inv_Z - 1, :])).argmin(0)
            all_alpha_dn[id_inv_Z, :] = alpha_dn[id_dn]
            id_up = np.abs(np.subtract.outer(alpha_up, all_alpha_up[id_inv_Z - 1, :])).argmin(0)
            all_alpha_up[id_inv_Z, :] = alpha_up[id_up]

        # return results
        if return_all:
            return all_alpha_dn, all_alpha_up
        else:
            return all_alpha_dn[-1, :], all_alpha_up[-1, :]

    def compute_radial_eigvals_hard(self, id_x, last_eigvals=None):
        """
        Compute the radial eigenvalues for a hard-walled duct with circular or annular cross section.

        :param int id_x: index that indicates the axial position at which the radial eigenvalues are computed
        :param array_like last_eigvals: initial guess of radial eigenvalues (None if not known)

        :returns: 1d numpy array containing the first nb_n radial eigenvalues
        :rtype: ndarray

        """
        # get attributes
        if self.type is 'cst':
            r1 = self.r1
            r2 = self.r2
        elif self.type is 'evol_x':
            r1 = self.r1[id_x]
            r2 = self.r2[id_x]
        m = self.m
        nb_n = self.nb_n

        # differentiate circular and annular cases
        ratio = r1 / r2
        if ratio < 1e-5:
            # if circular duct, jnp_zeros directly gives the radial eigenvalues
            list_alpha = list(sp.jnp_zeros(m, nb_n))
        else:
            # if annular duct, equation f = 0 is solved
            f = lambda r: sp.jvp(m, r, 1) * sp.yvp(m, r * ratio, 1) - sp.jvp(m, r * ratio, 1) * sp.yvp(m, r, 1)

            # find the first nb_n zeros of f
            if last_eigvals is None:
                # define the first interval
                xmin = 1e-6
                xmax = abs(m) + 1.5 * nb_n * pi
                resolution = 0.1
                counter = 0

                # find the roots inside this first interval
                list_alpha = find_real_roots(f, xmin, xmax, resolution)

                # if less than nb_n roots, the search is extended
                while len(list_alpha) < nb_n:
                    # update counter
                    counter += 1

                    if counter > 10:
                        # fill with 0 if some values are still missing
                        list_alpha.append(0.)
                        print('>>>>>>>>>>>>>>>> Warning: missed radial eigenvalue!')
                    else:
                        # otherwise, extend the search
                        # define new interval
                        xmin = xmax
                        xmax = xmax + 0.5 * nb_n * pi

                        # find the roots inside this new contour
                        list_alpha += find_real_roots(f, xmin, xmax, resolution)

                # keep only the first nb_n roots
                list_alpha = list_alpha[:nb_n]
            else:
                # if last position solutions are known, current position solutions are obtained directly by a Newton solver
                list_alpha = []
                success = 1
                for id_n in range(nb_n):
                    try:
                        alpha = opt.newton(f, last_eigvals[id_n] * r2, args=(), tol=1e-12, maxiter=20)
                        converged = True
                    except:
                        converged = False
                    if converged and (np.sum(np.abs(alpha - np.array(list_alpha)) < 1e-6) == 0):
                        success *= 1
                        list_alpha.append(alpha)
                    else:
                        success *= 0

                if success == 0:
                    list_alpha = self.compute_radial_eigvals_hard(id_x, last_eigvals=None)

        # normalize results
        list_alpha = [alpha / r2 for alpha in list_alpha]

        # return results
        return np.array(list_alpha)

    def compute_radial_eigvals_lined(self, id_x, prop_sign, inv_Z1=None, inv_Z2=None, last_eigvals=None):
        """
        Compute the radial eigenvalues for a lined duct with circular or annular cross section using the Delves and Lyness method.

        :param int id_x: index that indicates the axial position at which the radial eigenvalues are computed
        :param float prop_sign: propagation sign (-1. or 1.)
        :param complex inv_Z1: if not None, allows to force a value different form inv_Z1[id_x] for the inverse of hub impedance
        :param complex inv_Z2: if not None, allows to force a value different form inv_Z2[id_x] for the inverse of tip impedance
        :param array_like last_eigvals: initial guess of radial eigenvalues (None if not known)

        :returns: 1d numpy array containing the first nb_n radial eigenvalues
        :rtype: ndarray

        """
        # get attributes
        if self.type is 'cst':
            r1 = self.r1
            r2 = self.r2
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2
            D0 = self.D0
            U0 = self.U0
            C0 = self.C0
        elif self.type is 'evol_x':
            r1 = self.r1[id_x]
            r2 = self.r2[id_x]
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1[id_x]
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2[id_x]
            D0 = self.D0[id_x]
            U0 = self.U0[id_x]
            C0 = self.C0[id_x]
        w = self.w
        m = self.m
        nb_n = self.nb_n

        # differentiate circular and annular cases
        ratio = r1 / r2
        if ratio < 1e-5:
            # if circular duct, equation f = 0 is solved
            def f(r):
                # CHECK BRANCH SIGMA
                sigma2 = 1. - (C0 ** 2 - U0 ** 2) * r ** 2 / w ** 2
                sigma = np.where(np.sign(r.imag) >= 0., np.sqrt(sigma2) * prop_sign,
                                 np.conj(np.sqrt(sigma2) * prop_sign))
                Omega = w * C0 * (C0 - U0 * sigma) / (C0 ** 2 - U0 ** 2)
                xi2 = D0 * r2 / 1j / w * inv_Z2 * Omega ** 2
                fr = r * r2 * sp.jvp(m, r * r2, 1) - xi2 * sp.jv(m, r * r2)
                return fr

            def df(r):
                # CHECK BRANCH SIGMA
                sigma2 = 1. - (C0 ** 2 - U0 ** 2) * r ** 2 / w ** 2
                sigma = np.where(np.sign(r.imag) >= 0., np.sqrt(sigma2) * prop_sign,
                                 np.conj(np.sqrt(sigma2) * prop_sign))
                Omega = w * C0 * (C0 - U0 * sigma) / (C0 ** 2 - U0 ** 2)
                xi2 = D0 * r2 / 1j / w * inv_Z2 * Omega ** 2

                # d_sigma = -(C0 ** 2 - U0 ** 2) / w ** 2 * r / sigma
                d_sigma = -(C0 ** 2 - U0 ** 2) / w ** 2 * r / np.sqrt(sigma2)
                d_sigma = np.where(np.sign(r.imag) >= 0., d_sigma * prop_sign, np.conj(d_sigma * prop_sign))
                if np.size(r) > 1:
                    d_sigma = -np.where((np.abs(np.gradient(r).imag) > 0.) * (np.sign(r.imag) < 0.), d_sigma, -d_sigma)

                d_Omega = -w * C0 * (U0 * d_sigma) / (C0 ** 2 - U0 ** 2)
                d_xi2 = 2. * D0 * r2 / 1j / w * inv_Z2 * Omega * d_Omega
                d_fr = r * r2 ** 2 * sp.jvp(m, r * r2, 2) + r2 * (1. - xi2) * sp.jvp(m, r * r2, 1) - d_xi2 * sp.jv(m,
                                                                                                                   r * r2)
                return d_fr
        else:
            # if annular duct, equation f = 0 is solved
            def f(r):
                # CHECK BRANCH SIGMA
                sigma2 = 1. - (C0 ** 2 - U0 ** 2) * r ** 2 / w ** 2
                sigma = np.where(np.sign(r.imag) >= 0., np.sqrt(sigma2) * prop_sign,
                                 np.conj(np.sqrt(sigma2) * prop_sign))
                Omega = w * C0 * (C0 - U0 * sigma) / (C0 ** 2 - U0 ** 2)
                xi1 = D0 * r1 / 1j / w * inv_Z1 * Omega ** 2
                xi2 = D0 * r2 / 1j / w * inv_Z2 * Omega ** 2
                fr = (r * r2 * sp.jvp(m, r * r2, 1) - xi2 * sp.jv(m, r * r2)) * (r * r1 * sp.yvp(m, r * r1, 1) + xi1 * sp.yv(m, r * r1)) - \
                     (r * r2 * sp.yvp(m, r * r2, 1) - xi2 * sp.yv(m, r * r2)) * (r * r1 * sp.jvp(m, r * r1, 1) + xi1 * sp.jv(m, r * r1))
                return fr

            def df(r):
                # CHECK BRANCH SIGMA
                sigma2 = 1. - (C0 ** 2 - U0 ** 2) * r ** 2 / w ** 2
                sigma = np.where(np.sign(r.imag) >= 0., np.sqrt(sigma2) * prop_sign,
                                 np.conj(np.sqrt(sigma2) * prop_sign))
                Omega = w * C0 * (C0 - U0 * sigma) / (C0 ** 2 - U0 ** 2)
                xi1 = D0 * r1 / 1j / w * inv_Z1 * Omega ** 2
                xi2 = D0 * r2 / 1j / w * inv_Z2 * Omega ** 2

                # d_sigma = -(C0 ** 2 - U0 ** 2) / w ** 2 * r / sigma
                d_sigma = -(C0 ** 2 - U0 ** 2) / w ** 2 * r / np.sqrt(sigma2)
                d_sigma = np.where(np.sign(r.imag) >= 0., d_sigma * prop_sign, np.conj(d_sigma * prop_sign))
                if np.size(r) > 1:
                    d_sigma = -np.where((np.abs(np.gradient(r).imag) > 0.) * (np.sign(r.imag) < 0.), d_sigma, -d_sigma)

                d_Omega = -w * C0 * (U0 * d_sigma) / (C0 ** 2 - U0 ** 2)
                d_xi1 = 2. * D0 * r1 / 1j / w * inv_Z1 * Omega * d_Omega
                d_xi2 = 2. * D0 * r2 / 1j / w * inv_Z2 * Omega * d_Omega
                d_fr = (r * r2 ** 2 * sp.jvp(m, r * r2, 2) + r2 * (1. - xi2) * sp.jvp(m, r * r2, 1) - d_xi2 * sp.jv(m, r * r2)) * (r * r1 * sp.yvp(m, r * r1, 1) + xi1 * sp.yv(m, r * r1)) + \
                       (r * r2 * sp.jvp(m, r * r2, 1) - xi2 * sp.jv(m, r * r2)) * (r * r1 ** 2 * sp.yvp(m, r * r1, 2) + r1 * (1. + xi1) * sp.yvp(m, r * r1, 1) + d_xi1 * sp.yv(m, r * r1)) - \
                       (r * r2 ** 2 * sp.yvp(m, r * r2, 2) + r2 * (1. - xi2) * sp.yvp(m, r * r2, 1) - d_xi2 * sp.yv(m, r * r2)) * (r * r1 * sp.jvp(m, r * r1, 1) + xi1 * sp.jv(m, r * r1)) + \
                       (r * r2 * sp.yvp(m, r * r2, 1) - xi2 * sp.yv(m, r * r2)) * (r * r1 ** 2 * sp.jvp(m, r * r1, 2) + r1 * (1. + xi1) * sp.jvp(m, r * r1, 1) + d_xi1 * sp.jv(m, r * r1))
                return d_fr

        # find the first nb_n zeros of f
        if last_eigvals is None:
            # use of the Delves and Lyness method coupled with a Newton solver if last position solutions are not known
            # define the first contour limits
            xmin = 0.75 * abs(m) / r2
            xmax = (abs(m) + 1.25 * nb_n * pi) / r2
            ymin = (-2. * pi - 1e-6) / r2
            ymax = (2. * pi) / r2
            resolution = 0.005 / r2
            counter = 0

            # find the roots inside this first contour
            list_alpha = find_complex_roots(f, xmin, xmax, ymin, ymax, resolution, 0.1 * resolution, df=None,
                                                max_loops=50)

            # if less than nb_n roots, the search is extended
            while len(list_alpha) < nb_n:
                # update counter
                counter += 1

                if counter > 10:
                    # fill with 0 if some values are still missing
                    list_alpha.append(0.)
                    print('>>>>>>>>>>>>>>>> Warning: missed radial eigenvalue!')
                else:
                    # otherwise, extend the search
                    # define new contour
                    if len(list_alpha) <= 2:
                        xmin = xmax
                        xmax = xmax + (1.25 * (nb_n - len(list_alpha)) * pi) / r2
                    else:
                        mean_dalpha = np.mean(np.array(list_alpha[1:]) - np.array(list_alpha[:-1])).real
                        xmin = list_alpha[-1].real + mean_dalpha / 2.
                        xmax = list_alpha[-1].real + mean_dalpha / 2. + mean_dalpha * (nb_n - len(list_alpha))

                    # find the roots inside this new contour
                    list_alpha += utnum.find_complex_roots(f, xmin, xmax, ymin, ymax, resolution, 0.1 * resolution,
                                                         df=None, max_loops=50)

            # sort results with increasing squared amplitude
            list_alpha = [alpha for alpha in sorted(list_alpha, key=lambda x: x.real ** 2 + x.imag ** 2)]
        else:
            # if last position solutions are known, current position solutions are obtained directly by a Newton solver
            list_alpha = []
            success = 1
            for id_n in range(nb_n):
                try:
                    alpha = opt.newton(f, last_eigvals[id_n], fprime=None, args=(), tol=1e-12, maxiter=20) # check df definition
                    converged = True
                except:
                    converged = False
                if converged and (np.sum(np.abs(alpha - np.array(list_alpha)) < 1e-6) == 0):
                    success *= 1
                    list_alpha.append(alpha)
                else:
                    success *= 0
            if success == 0:
                list_alpha = self.compute_radial_eigvals_lined(id_x, prop_sign, inv_Z1, inv_Z2, last_eigvals=None)

        # return results
        return np.array(list_alpha)

    def get_radial_eigmods(self, id_x, alpha, prop_sign, inv_Z1=None, inv_Z2=None, norm='surf'):
        """
        Compute the radial evolution of eigenmodes.

        :param int id_x: index that indicates the axial position at which the eigenmodes are evaluated
        :param array_like alpha: radial eigenvalues
        :param float prop_sign: propagation sign (-1. or 1.)
        :param complex inv_Z1: if not None, allows to force a value different form inv_Z1[id_x] for the inverse of hub impedance
        :param complex inv_Z2: if not None, allows to force a value different form inv_Z2[id_x] for the inverse of tip impedance
        :param str norm: kind of normalization ('surf', 'tip')

        :returns: radial eigenmode functions
        :rtype: func

        """
        # get attributes
        # get attributes
        if self.type is 'cst':
            r1 = self.r1
            r2 = self.r2
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2
            D0 = self.D0
            U0 = self.U0
            C0 = self.C0
        elif self.type is 'evol_x':
            r1 = self.r1[id_x]
            r2 = self.r2[id_x]
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1[id_x]
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2[id_x]
            D0 = self.D0[id_x]
            U0 = self.U0[id_x]
            C0 = self.C0[id_x]
        w = self.w
        m = self.m

        # function to manage shapes
        reshape = lambda r: np.array(r).reshape(
            (np.array(r).shape + tuple([1 for i in range(len(np.array(alpha).shape))])))

        # compute useful variables
        sigma2 = (1. - (C0 ** 2 - U0 ** 2) * alpha ** 2 / w ** 2).astype(complex)
        sigma = prop_sign * np.where(np.sqrt(sigma2).imag >= 1e-8, -np.sqrt(sigma2), np.sqrt(sigma2))
        Omega = w * C0 * (C0 - U0 * sigma) / (C0 ** 2 - U0 ** 2)
        xi1 = Omega ** 2 * D0 * r1 / 1j / w * inv_Z1
        xi2 = Omega ** 2 * D0 * r2 / 1j / w * inv_Z2

        # circular case
        if r1 / r2 < 1e-5:
            # compute eigenmode (psi)
            psi = lambda r: sp.jv(m, alpha * reshape(r))

            # compute eigenmode norm (norm)
            if norm is 'surf':
                norm = np.sqrt(0.5 * pi * r2 ** 2 * (1. - (m ** 2 - xi2 ** 2) / (alpha * r2) ** 2) * sp.jv(m, alpha * r2) ** 2)
                if m == 0: norm *= 2.
            elif norm is 'tip':
                norm = psi(r2)

            # compute normalized eigenmode (psi_norm)
            psi_norm = lambda r: psi(r) / norm

        # annular case
        else:
            # compute -M/N factor (fact)
            fact = (alpha * r1 * sp.jvp(m, alpha * r1, 1) + xi1 * sp.jv(m, alpha * r1)) / \
                   (alpha * r1 * sp.yvp(m, alpha * r1, 1) + xi1 * sp.yv(m, alpha * r1))

            # compute eigenmode (psi)
            psi = lambda r: sp.jv(m, alpha * reshape(r)) - fact * sp.yv(m, alpha * reshape(r))

            # compute eigenmode norm (norm)
            if norm == 'surf':
                norm = np.sqrt(0.5 * pi * r2 ** 2 * (1. - (m ** 2 - xi2 ** 2) / (alpha * r2) ** 2) *
                                   (sp.jv(m, alpha * r2) - fact * sp.yv(m, alpha * r2)) ** 2 - \
                               0.5 * pi * r1 ** 2 * (1. - (m ** 2 - xi1 ** 2) / (alpha * r1) ** 2) *
                                   (sp.jv(m, alpha * r1) - fact * sp.yv(m, alpha * r1)) ** 2)
                if m == 0: norm *= 2.
            elif norm == 'tip':
                norm = psi(r2)

            # compute normalized eigenmode (psi_norm)
            psi_norm = lambda r: psi(r) / norm

        # return normalized eigenmode
        return psi_norm

    def get_radial_eigmods_derivative(self, id_x, alpha, prop_sign, inv_Z1=None, inv_Z2=None, norm='surf'):
        """
        Compute the radial evolution of the radial derivative of the eigenmodes.

        :param int id_x: index that indicates the axial position at which the eigenmodes are evaluated
        :param array_like alpha: radial eigenvalues
        :param float prop_sign: propagation sign (-1. or 1.)
        :param complex inv_Z1: if not None, allows to force a value different form inv_Z1[id_x] for the inverse of hub impedance
        :param complex inv_Z2: if not None, allows to force a value different form inv_Z2[id_x] for the inverse of tip impedance
        :param str norm: kind of normalization ('surf', 'tip')

        :returns: radial eigenmode functions
        :rtype: func

        """
        # get attributes
        # get attributes
        if self.type is 'cst':
            r1 = self.r1
            r2 = self.r2
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2
            D0 = self.D0
            U0 = self.U0
            C0 = self.C0
        elif self.type is 'evol_x':
            r1 = self.r1[id_x]
            r2 = self.r2[id_x]
            if inv_Z1 is None:
                inv_Z1 = self.inv_Z1[id_x]
            if inv_Z2 is None:
                inv_Z2 = self.inv_Z2[id_x]
            D0 = self.D0[id_x]
            U0 = self.U0[id_x]
            C0 = self.C0[id_x]
        w = self.w
        m = self.m

        # function to manage shapes
        reshape = lambda r: np.array(r).reshape(
            (np.array(r).shape + tuple([1 for i in range(len(np.array(alpha).shape))])))

        # compute useful variables
        sigma2 = (1. - (C0 ** 2 - U0 ** 2) * alpha ** 2 / w ** 2).astype(complex)
        sigma = prop_sign * np.where(np.sqrt(sigma2).imag >= 1e-8, -np.sqrt(sigma2), np.sqrt(sigma2))
        Omega = w * C0 * (C0 - U0 * sigma) / (C0 ** 2 - U0 ** 2)
        xi1 = Omega ** 2 * D0 * r1 / 1j / w * inv_Z1
        xi2 = Omega ** 2 * D0 * r2 / 1j / w * inv_Z2

        # circular case
        if r1 / r2 < 1e-5:
            # compute eigenmode (psi)
            psi = lambda r: alpha * sp.jvp(m, alpha * reshape(r))

            # compute eigenmode norm (norm)
            if norm is 'surf':
                norm = np.sqrt(0.5 * pi * r2 ** 2 * (1. - (m ** 2 - xi2 ** 2) / (alpha * r2) ** 2) * sp.jv(m, alpha * r2) ** 2)
                if m == 0: norm *= 2.
            elif norm is 'tip':
                norm = psi(r2)

            # compute normalized eigenmode (psi_norm)
            psi_norm = lambda r: psi(r) / norm

        # annular case
        else:
            # compute -M/N factor (fact)
            fact = (alpha * r1 * sp.jvp(m, alpha * r1, 1) + xi1 * sp.jv(m, alpha * r1)) / \
                   (alpha * r1 * sp.yvp(m, alpha * r1, 1) + xi1 * sp.yv(m, alpha * r1))

            # compute eigenmode (psi)
            psi = lambda r: alpha * sp.jvp(m, alpha * reshape(r)) - fact * alpha * sp.yvp(m, alpha * reshape(r))

            # compute eigenmode norm (norm)
            if norm == 'surf':
                norm = np.sqrt(0.5 * pi * r2 ** 2 * (1. - (m ** 2 - xi2 ** 2) / (alpha * r2) ** 2) *
                                   (sp.jv(m, alpha * r2) - fact * sp.yv(m, alpha * r2)) ** 2 - \
                               0.5 * pi * r1 ** 2 * (1. - (m ** 2 - xi1 ** 2) / (alpha * r1) ** 2) *
                                   (sp.jv(m, alpha * r1) - fact * sp.yv(m, alpha * r1)) ** 2)
                if m == 0: norm *= 2.
            elif norm == 'tip':
                norm = psi(r2)

            # compute normalized eigenmode (psi_norm)
            psi_norm = lambda r: psi(r) / norm

        # return normalized eigenmode
        return psi_norm


