from numpy.lib.scimath import sqrt
from scipy.spatial.distance import pdist, squareform
from math import pi
import numpy as np
import scipy.integrate as it
import scipy.optimize as opt
import scipy.special as sp


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

