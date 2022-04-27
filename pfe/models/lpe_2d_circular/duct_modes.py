"""Duct mode boundary condition for the LPE in 2D"""
from numpy.lib.scimath import sqrt
from scipy.spatial.distance import pdist, squareform
from math import pi
import numpy as np
import scipy.integrate as it
import scipy.optimize as opt
import scipy.special as sp
from .Bessel import Bessel


def gradx(a, dx, order=2):
    """
    Compute the gradient of a 2D array along its first direction.

    :param array_like a: 2D array over which the x-gradient is computed
    :param float dx: mesh step between the elements of the array along its first dimension
    :param int order: order of the gradient computation (2, 5 or 6)

    :returns: x-gradient array with the same shape as the input array
    :rtype: ndarray

    """
    # initialize with 2nd-order gradient computation
    np_ver = np.version.version
    np_ver = (int(np_ver.split("'")[-1].split('.')[0]), int(np_ver.split('.')[1]), int(np_ver.split('.')[2]))
    if (np_ver[0] > 1) or (np_ver[0] == 1 and np_ver[1] >= 11):
        # for newer numpy versions, direct computation along x-axis using the axis argument
        da_dx = np.gradient(a, dx, axis=0)
    else:
        # for older numpy versions, the computation is done for each radial line (no axis argument)
         da_dx = np.zeros_like(a, dtype='complex')
         for id_r in range(a.shape[1]):
            da_dx[:, id_r] = np.gradient(a[:, id_r], dx)

    # re-compute the interior points with higher order if specified
    if order == 5:
        da_dx[5:-5, :] = (4. * a[:-10, :] - 25. * a[1:-9, :] + 600. * a[3:-7, :] - 4200. * a[4:-6, :] - \
                            2520. * a[5:-5, :] + 8400. * a[6:-4, :] - 3000. * a[7:-3, :] + 900. * a[8:-2, :] - \
                            175. * a[9:-1, :] + 16. * a[10:, :]) / (7560. * dx)
    elif order == 6:
        da_dx[6:-6, :] = (-5. * a[:-12, :] + 48. * a[1:-11, :] - 165. * a[2:-10, :] + 2475. * a[4:-8, :] - \
                            15840. * a[5:-7, :] - 9240. * a[6:-6, :] + 31680. * a[7:-5, :] - 12375. * a[8:-4, :] + \
                            4400. * a[9:-3, :] - 1155. * a[10:-2, :] + 192. * a[11:-1, :] - 15. * a[12:, :]) / (27720. * dx)

    # return results
    return da_dx

def cumint(a, dx, dir):
    """
    Compute the cumulative integral of a 1D array.

    :param array_like a: 1D array over which the cumulative integral is computed
    :param float dx: mesh step between the elements of the array
    :param float dir: define the direction of the cumulative integral (-1. for upstream direction, 1. for downstream direction)

    :returns: cumulative integral array with the same shape as the input array
    :rtype: ndarray

    """
    # initialize the cumulative integral
    cumint = np.zeros_like(a, dtype=a.dtype)

    # compute the cumulative integral
    if dir == 1.:
        # case of downstream integral
        a_cell = 0.5 * (a[:-1] + a[1:])
        cumint[1:] = np.cumsum(a_cell, axis=0) * dx
    elif dir == -1.:
        # case of upstream integral
        a_reverse = a[::-1]
        a_cell = 0.5 * (a_reverse[:-1] + a_reverse[1:])
        cumint[1:] = np.cumsum(a_cell, axis=0) * dx
        cumint = -cumint[::-1]
    else:
        raise ValueError("dir must be in [-1., 1.]!")

    # return results
    return cumint

def find_real_roots(f, xmin, xmax, resolution):
    """
    Find the zeros of the real function f in the interval [xmin, xmax] using a bisection method.

    :param func f: function whose zeros are to be found
    :param float xmin: lower limit of the searching interval
    :param float xmax: upper limit of the searching interval
    :param float resolution: resolution of the box

    :returns: list containing the zeros of f in the interval [xmin, xmax]
    :rtype: list

    """
    # roughly find the nb_n first zeros of f detecting sign changes
    x = np.arange(xmin, xmax, resolution)
    f_x = f(x)
    id_sign_change = np.where(f_x[1:] * f_x[:-1] < 0.)
    list_roots_low = list(x[id_sign_change])
    list_roots_upp = list(x[id_sign_change[0] + 1])

    # make the solutions more accurate using a bisection method
    list_roots = []
    for k in range(len(list_roots_low)):
        root, out = opt.bisect(f, list_roots_low[k], list_roots_upp[k], full_output=True)
        assert(out.converged is True)
        list_roots.append(out.root)

    # return list of roots
    return list_roots

def find_complex_roots(f, xmin, xmax, ymin, ymax, init_resolution, min_resolution, df=None, max_loops=50):
    """
    Find the zeros of the complex function f in the box [xmin, xmax] x [ymin, ymax] using the Delves and Lyness method coupled with a Newton solver.

    :param func f: function whose zeros are to be found
    :param float xmin: lower limit of the real part searching interval
    :param float xmax: upper limit of the real part searching interval
    :param float ymin: lower limit of the imaginary part searching interval
    :param float ymax: upper limit of the imaginary part searching interval
    :param float init_resolution: initial resolution of the box
    :param float min_resolution: minimal resolution of the box
    :param func df: derivative of f (None if not known)
    :param int max_loops: maximum number of recursive searches

    :returns: list containing the zeros of f in the box [xmin, xmax] x [ymin, ymax]
    :rtype: list

    """
    # function that create box points as a 2D array
    def create_box(xmin, xmax, ymin, ymax, resolution):
        # get discretization
        nbx = int(ceil((xmax - xmin) / resolution))
        nby = int(ceil((ymax - ymin) / resolution))

        # create box points as a 2D array
        box_points = np.zeros((2 * (nbx + nby), 2))
        box_points[0:nbx, 0] = np.linspace(xmin, xmax, nbx, endpoint=False)
        box_points[nbx:(nbx + nby), 0] = np.linspace(xmax, xmax, nby, endpoint=False)
        box_points[(nbx + nby):(2 * nbx + nby), 0] = np.linspace(xmax, xmin, nbx, endpoint=False)
        box_points[(2 * nbx + nby):2 * (nbx + nby), 0] = np.linspace(xmin, xmin, nby)
        box_points[0:nbx, 1] = np.linspace(ymin, ymin, nbx, endpoint=False)
        box_points[nbx:(nbx + nby), 1] = np.linspace(ymin, ymax, nby, endpoint=False)
        box_points[(nbx + nby):(2 * nbx + nby), 1] = np.linspace(ymax, ymax, nbx, endpoint=False)
        box_points[(2 * nbx + nby):2 * (nbx + nby), 1] = np.linspace(ymax, ymin, nby)

        # return box points
        return box_points

    # function that counts the number of roots inside the given box using the Cauchy argument
    def count_roots_inside_box(f, df, box_points):
        # compute f and df on all box points
        box_points_cmplx = box_points[:, 0] + 1j * box_points[:, 1]
        f_box = f(box_points_cmplx)
        if df is None:
            df_box = np.gradient(f_box, box_points_cmplx)
        else:
            df_box = df(box_points_cmplx)

        # integration using the composite Simpson's rule to get the Cauchy argument
        Z = it.simps(df_box / f_box, x=box_points_cmplx) / (2. * pi * 1j)

        # get estimated number of roots and error
        nb_roots = int(round(Z.real))
        error = abs(Z - nb_roots)

        # return results
        return nb_roots, error, [f_box, df_box]

    # function that computes the Cauchy argument of order k
    def cauchy_argument_k(box_points, k, store):
        # get stored results
        f_box = store[0]
        df_box = store[1]

        # compute Cauchy argument of order k
        box_points_cmplx = box_points[:, 0] + 1j * box_points[:, 1]
        p_k = it.simps(df_box / f_box * pow(box_points_cmplx, k), x=box_points_cmplx) / (2. * pi * 1j)

        # return result
        return p_k

    # recursive root finding algorithm
    def recursive_root_finding(f, df, xmin, xmax, ymin, ymax, resolution):
        # update counter
        counter[0] += 1

        if counter[0] <= max_loops:
            # count roots inside the box
            refine = True
            while refine:
                box_points = create_box(xmin, xmax, ymin, ymax, resolution)
                nb_roots, error, store = count_roots_inside_box(f, df, box_points)
                if (error < 1e-3) or (resolution < min_resolution):
                    refine = False
                else:
                    resolution /= 2.

            nb_known_roots = np.sum((xmin < np.array(list_roots).real) * (np.array(list_roots).real < xmax) * \
                                    (np.array(list_roots).imag < ymax) * (ymin < np.array(list_roots).imag))
            nb_roots_to_find = nb_roots - nb_known_roots

            if nb_roots_to_find == 0:
                # no root inside the box
                pass
            elif nb_roots_to_find <= 10:
                # less than ten roots to find in the box
                # trying to find them with the Delves and Lyness method
                if (xmax - xmin) < (10. * resolution):
                    # the box is too small and contains one or several roots
                    # we just add the middle box value as the solution(s)
                    # this case can happen when two roots collapse (or when their curves interset)
                    for k in range(nb_roots_to_find):
                        root = 0.5 * (xmin + xmax) + 1j * 0.5 * (ymin + ymax)
                        list_roots.append(root)
                        print('>>>>>>>>>>>>>>>> Warning: approximated root!')
                else:
                    list_pk = [nb_roots]
                    list_ek = [1.]
                    list_poly_coeff = [1.]
                    for k in range(1, nb_roots + 1):
                        list_pk.append(cauchy_argument_k(box_points, k, store))
                        ek = 0.
                        for i in range(1, k + 1):
                            ek += (1. / k) * (-1.) ** (i - 1) * list_ek[k-i] * list_pk[i]
                        list_ek.append(ek)
                        list_poly_coeff.append((-1.) ** k * ek)
                    list_roots_approx = poly.polyroots(list_poly_coeff[::-1])
                    nb_found_roots = 0
                    for k in range(nb_roots):
                        try:
                            root = opt.newton(f, list_roots_approx[k], fprime=df, args=(), tol=1e-12, maxiter=20)
                            converged = True
                        except:
                            converged = False
                        if converged and root.real >= xmin and root.real < xmax and root.imag >= ymin and root.imag < ymax and (np.sum(np.abs(root - np.array(list_roots)) < 1e-6) == 0):
                            list_roots.append(root)
                            nb_found_roots += 1
                    if nb_found_roots != nb_roots_to_find:
                        xmid = 0.5 * (xmin + xmax)
                        ymid = 0.5 * (ymin + ymax)
                        recursive_root_finding(f, df, xmin, xmid, ymin, ymid, resolution)
                        recursive_root_finding(f, df, xmid, xmax, ymin, ymid, resolution)
                        recursive_root_finding(f, df, xmin, xmid, ymid, ymax, resolution)
                        recursive_root_finding(f, df, xmid, xmax, ymid, ymax, resolution)
            else:
                # too much roots for Delves and Lyness method, the box is divided into four subboxes
                xmid = 0.5 * (xmin + xmax)
                ymid = 0.5 * (ymin + ymax)
                recursive_root_finding(f, df, xmin, xmid, ymin, ymid, resolution)
                recursive_root_finding(f, df, xmid, xmax, ymin, ymid, resolution)
                recursive_root_finding(f, df, xmin, xmid, ymid, ymax, resolution)
                recursive_root_finding(f, df, xmid, xmax, ymid, ymax, resolution)
        else:
            print('>>>>>>>>>>>>>>>> Warning: some roots may have been missed!')

    # initialize counter and list of roots
    list_roots = []
    counter = [0]

    # launch recursive root finding algorithm
    recursive_root_finding(f, df, xmin, xmax, ymin, ymax, init_resolution)

    # sort roots with increasing squared amplitude
    list_roots = [root for root in sorted(list_roots, key=lambda x: x.real ** 2 + x.imag ** 2)]

    # return list of roots
    return list_roots




class Basis:
    """Modal basis for a 2D duct"""

    def __init__(self):
        """Constructor"""
        self.num_modes = None
        self.c0 = None
        self.u0n = None
        self.normal = None
        self.xc = None
        self.azimuthal_mode = 0
        self.r2 = 1
        self.H = None

    def xy_to_tau(self, x, y):
        """Convert physical coordinates to surface coordinate

        :param x: Physical coordinate
        :type x: A scalar or a Numpy array
        :param y: Physical coordinate
        :type y: A scalar or a Numpy array
        :return: The coordinate on the surface
        :rtype: A scalar or a Numpy array
        """
        tau = (
                   self.H / 2
                   - (x - self.xc[0]) * self.normal[1]
        + (y - self.xc[1]) * self.normal[0]
        )
        return tau

    def phi(self, bessel):
        """Mode shape functions

        :param bessel: Bessel class
        :type Class: a class
        :return: Mode shape functions in the two propagation directions
        :rtype: A tuple of two Numpy arrays
        """
        nb_n = np.arange(self.num_modes)[-1]
        bessel.nb_n = nb_n+1
        alpha = bessel.compute_radial_eigvals_hard(0)
        psi = bessel.get_radial_eigmods(0,alpha,1)
        return (psi, psi)

    def dphidn(self, bessel, omega):
        """Normal derivatives of the mode shape functions

        :param bessel: Bessel class
        :type bessel: a class
        :param omega: angular frequency
        :type omega: A scalar
        :return: Mode shape functions in the two propagation directions
        :rtype: A tuple of two Numpy arrays
        """

        nb_n = np.arange(self.num_modes)[-1]
        bessel.nb_n = nb_n+1
        alpha = bessel.compute_radial_eigvals_hard(0)
        psi = bessel.get_radial_eigmods(0,alpha,1)

        k_o = np.conj(
            (
                self.c0 * sqrt(omega ** 2 - (self.c0 ** 2 - self.u0n ** 2) * alpha ** 2)
                - self.u0n * omega
            )
            / (self.c0 ** 2 - self.u0n ** 2)
        )
        k_i = np.conj(
            (
                -self.c0
                * sqrt(omega ** 2 - (self.c0 ** 2 - self.u0n ** 2) * alpha ** 2)
                - self.u0n * omega
            )
            / (self.c0 ** 2 - self.u0n ** 2)
        )

        psi_o = lambda r: -1j * k_o[None, :] * psi(r)
        psi_i = lambda r: -1j * k_i[None, :] * psi(r)
        return (psi_o, psi_i)

    def dphidtau(self, bessel):
        """Tangential derivatives of the mode shape functions

        :param bessel: Bessel class
        :type bessel: a class
        :return: Mode shape functions in the two propagation directions
        :rtype: A tuple of two Numpy arrays
        """

        nb_n = np.arange(self.num_modes)[-1]
        bessel.nb_n = nb_n+1
        alpha = bessel.compute_radial_eigvals_hard(0)
        psi = bessel.get_radial_eigmods_derivative(0,alpha,1)

        return (psi, psi)

    def norms(self):
        """Norms of the mode shape functions

        :return: The norms of the outgoing and incoming mode shape functions
        :rtype: A tuple of two Numpy arrays
        """
        Q = np.ones((self.num_modes,))
        N = np.diag(Q * self.H)
        return (N, N)



class DuctModes:
    """Duct mode boundary condition for the LPE in 2D"""

    def __init__(self, domain, modes_o, modes_i=None):
        """Constructor

        :param domain: The finite-element model
        :type domain: An instance of pfe.Model
        :param Ao: Amplitude of the outgoing modes
        :type Ao: An instance of pfe.Vector
        :param Ai: Amplitude of the incoming modes, defaults to None
        :type Ai: A sequence of scalars, optional
        """
        self.domain = domain
        self.modes_o = modes_o
        if modes_i is not None:
            self.modes_i = np.asarray(modes_i)
        else:
            self.modes_i = None
        self.basis = Basis()
        # Get the normal vector from the first element
        e = self.domain.get_elements(dim=1)[0]
        self.basis.normal = self.domain.element_geometry(e).normal(np.array([[0.0]]))[0]
        # Compute the width and center of the boundary
        # We assume that the surface is flat
        nodes = self.domain.nodes()
        D = squareform(pdist(nodes.T))
        self.basis.H = 1
        n1, n2 = np.unravel_index(np.argmax(D), D.shape)
        self.basis.xc = (nodes[:, n1] + nodes[:, n2]) / 2

    def assemble(self, model, system):
        """Assemble the element matrices

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param system: The algebraic system ton contribute to
        :type system: An instance of a class from pfe.algebra
        """
        elements = self.domain.get_elements(dim=1)
        # Set the number of modes to use
        self.basis.num_modes = model.fields[self.modes_o].length
        # Compute the mean sound speed and flow normal velocity
        u0fun = model.parameters["u0"]
        v0fun = model.parameters["v0"]
        c0fun = model.parameters["c0"]
        quad_order = u0fun.order
        u0 = 0.0
        v0 = 0.0
        c0 = 0.0
        L = 0.0
        for element in elements:
            geometry = self.domain.element_geometry(element)
            u, weights = geometry.integration(quad_order)
            u0 += weights.dot(u0fun.get_value(element, u))
            v0 += weights.dot(v0fun.get_value(element, u))
            c0 += weights.dot(c0fun.get_value(element, u))
            L += np.sum(weights)
        self.basis.c0 = c0 / L
        self.basis.u0n = (u0 * self.basis.normal[0] + v0 * self.basis.normal[1]) / L
        # Assemble the element contributions
        for element in elements:
            K12, K21, F1 = self.terms(model, element)
            dof1 = model.fields["phi"].element_dofs(element)
            dof2 = model.fields[self.modes_o].dofs
            system.lhs.add_matrix(dof1, dof2, K12)
            system.lhs.add_matrix(dof2, dof1, K21)
            if F1 is not None:
                system.rhs.add(dof1, dof1 * 0, F1.flatten())


        Q_oo, Q_io = self.basis.norms()
        K22 = Q_oo.copy()
        system.lhs.add_matrix(dof2, dof2, K22)
        if self.modes_i is not None:
            F2 = -Q_io @ self.modes_i
            system.rhs.add(dof2, dof2 * 0, F2.flatten())

    def terms(self, model, e):
        """Compute the element matrices for a single element

        :param model: The finite-element model
        :type model: An instance of pfe.Model
        :param e: The element tag
        :type e: A positive integer
        :return: The element matrices
        :rtype: A tuple of Numpy arrays
        """
        basis = model.fields["phi"].basis(e)
        geometry = self.domain.element_geometry(e)
        quad_order = basis.order + 3
        u, weights = geometry.integration(quad_order)
        xr = geometry.position(u)
        x, r = xr
        tau = geometry.tangent(u)
        n = geometry.normal(u)
        phi, _ = geometry.basis_from_order(basis, quad_order)
        m = model.parameters["m"].get_value()
        omega = model.parameters["omega"].get_value()
        r1 = model.parameters["r1"].get_value(e, u, xr)[0]
        r2 = model.parameters["r2"].get_value(e, u, xr)[0]
        u0 = model.parameters["u0"].get_value(e, u, xr)
        v0 = model.parameters["v0"].get_value(e, u, xr)
        rho0 = model.parameters["rho0"].get_value(e, u, xr)
        c0 = model.parameters["c0"].get_value(e, u, xr)
        u0tau = u0 * tau[:, 0] + v0 * tau[:, 1]
        u0n = u0 * n[:, 0] + v0 * n[:, 1]

        bessel = Bessel(type='cst',x=x,r1=r1,r2=r2,inv_Z1=0,inv_Z2=0, U0=np.mean(u0),
                        C0=np.mean(c0), D0=np.mean(rho0), w= omega, m=m)
        phi_o, phi_i = self.basis.phi(bessel)
        phi_o, phi_i = 1/np.sqrt(2) * phi_o(r), 1/np.sqrt(2) * phi_i(r)
        dphidn_o, dphidn_i = self.basis.dphidn(bessel,omega)
        dphidn_o, dphidn_i = 1/np.sqrt(2) * dphidn_o(r), 1/np.sqrt(2) * dphidn_i(r)
        dphidtau_o, dphidtau_i = self.basis.dphidtau(bessel)
        dphidtau_o, dphidtau_i = 1/np.sqrt(2) * dphidtau_o(r), 1/np.sqrt(2) * dphidtau_i(r)

        if self.modes_i is not None:
            F1 = (
                phi.T @ ((1j * omega * weights * rho0 * u0n * r * 2 * np.pi / c0 ** 2)[:, None] * phi_i)
                + phi.T
                @ ((weights * rho0 * u0n * u0tau * r * np.pi * 2 / c0 ** 2)[:, None] * dphidtau_i)
                - phi.T @ ((weights * rho0 * r * np.pi * 2 * (1 - (u0n / c0) ** 2))[:, None] * dphidn_i)
            ) @ self.modes_i
        else:
            F1 = None
        K12 = (
            -phi.T @ ((1j * omega * weights * rho0 * u0n * r * np.pi * 2 / c0 ** 2)[:, None] * phi_o)
            - phi.T @ ((weights * rho0 * u0n * u0tau * r * np.pi * 2 / c0 ** 2)[:, None] * dphidtau_o)
            + phi.T @ ((weights * rho0 * r * np.pi * 2 * (1 - (u0n / c0) ** 2))[:, None] * dphidn_o)
        )
        K21 = -phi_o.T @ ((weights * r * np.pi * 2 )[:, None] * phi)


        return (K12, K21, F1)
