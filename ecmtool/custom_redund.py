from time import time

from mpi4py import MPI
import numpy as np
import os
from scipy.linalg import LinAlgError
from scipy.optimize import linprog

from ecmtool.helpers import mp_print
try:
    from ecmtool._bglu_dense import BGLU
except (ImportError, EnvironmentError, OSError):
    from ecmtool.bglu_dense_uncompiled import BGLU
from ecmtool.helpers import redund, get_metabolite_adjacency, to_fractions
from ecmtool.intersect_directly_mpi import perturb_LP, normalize_columns, independent_rows, get_start_basis,\
    add_first_ray


def kkt_check_redund(c, A, x, basis, i, tol=1e-8, threshold=1e-3, max_iter=100000, verbose=True):
    """
    Determine whether KKT conditions hold for x0.
    Take size 0 steps if available.
    """
    improvement = False
    init_actives = [i]
    ab = np.arange(A.shape[0])
    a = np.arange(A.shape[1])

    maxupdate = 10
    B = BGLU(A, basis, maxupdate, False)
    iteration = 0
    while True:
        bl = np.zeros(len(a), dtype=bool)
        bl[basis] = 1
        xb = x[basis]

        try:
            l = B.solve(c[basis], transposed=True)  # similar to v = linalg.solve(B.T, c[basis])
        except LinAlgError:
            np.set_printoptions(threshold=np.inf)
            mp_print('This matrix seems to be singular:', PRINT_IF_RANK_NONZERO=True)
            mp_print(B.B, PRINT_IF_RANK_NONZERO=True)
            mp_print('Iteration:' + str(iteration), PRINT_IF_RANK_NONZERO=True)
            mp_print('u:', PRINT_IF_RANK_NONZERO=True)
            mp_print(u, PRINT_IF_RANK_NONZERO=True)
            print("LinAlgError in B.solve")
            np.set_printoptions(threshold=1000)
            return True, 1

        sn = c - l.dot(A)  # reduced cost
        sn = sn[~bl]

        if np.all(sn >= -tol):  # in this case x is an optimal solution
            return True, 0

        entering = a[~bl][np.argmin(sn)]
        u = B.solve(A[:, entering])

        i = u > tol  # if none of the u are positive, unbounded
        if not np.any(i):
            mp_print("Warning: unbounded problem in KKT_check")
            return True, 1

        th = xb[i] / u[i]
        l = np.argmin(th)  # implicitly selects smallest subscript
        if basis[i][l] in init_actives:  # if either plus or minus leaves basis, LP has made significant improvement
            improvement = True

        step_size = th[l]  # step size

        # Do pivot
        x[basis] = x[basis] - step_size * u
        x[entering] = step_size
        x[abs(x) < 10e-20] = 0
        B.update(ab[i][l], entering)  # modify basis
        basis = B.b

        # if np.dot(c, x) < -threshold:  # found a better solution, so not adjacent
        if improvement:
            if not np.dot(c, x) < -threshold:
                mp_print('Original way of finding non-adjacents does not say these are non-adjacent', True)
            # if verbose:
            #     mp_print("Did %d steps in kkt_check, found False - c*x %.8f" % (iteration, np.dot(c, x)))
            return False, 0

        iteration += 1
        if iteration % 10000 == 0:
            print("Warning: reached %d iterations" % iteration)


def setup_LP_redund(R_indep, i):
    number_rays = R_indep.shape[1]
    b_eq = R_indep[:, i]
    c = -np.ones(number_rays)
    c[i] = 0
    x0 = np.zeros(number_rays)
    x0[i] = 1

    return R_indep, b_eq, c, x0


def check_extreme(R, i, basis, tol=1e-10):
    A_eq, b_eq, c, x0 = setup_LP_redund(R, i)
    b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, tol)
    KKT, status = kkt_check_redund(c, A_eq, x0, basis, i)

    if status == 0:
        return KKT
    else:
        mp_print('LinAlgError in an adjacency test. Check if this happens more often.', PRINT_IF_RANK_NONZERO=True)
        mp_print('Now assuming that rays are adjacent.', PRINT_IF_RANK_NONZERO=True)
        return True


def drop_redundant_rays(ray_matrix):
    matrix_normalized = normalize_columns(ray_matrix)
    matrix_indep_rows = independent_rows(matrix_normalized)

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    # first find any column basis of R_indep
    start_basis = get_start_basis(matrix_indep_rows)
    start_basis_inv = np.linalg.inv(matrix_indep_rows[:, start_basis])

    number_rays = matrix_indep_rows.shape[1]
    non_extreme_rays = []
    for i in range(number_rays):
        if i % mpi_size == mpi_rank:
            basis = add_first_ray(matrix_indep_rows, start_basis_inv, start_basis, i)
            extreme = check_extreme(matrix_indep_rows, i, basis)
            if not extreme:
                non_extreme_rays.append(i)

    # MPI communication step
    non_extreme_sets = comm.allgather(non_extreme_rays)
    for i in range(mpi_size):
        if i != mpi_rank:
            non_extreme_rays.extend(non_extreme_sets[i])
    non_extreme_rays.sort()

    return np.delete(ray_matrix, non_extreme_rays, axis=1)
