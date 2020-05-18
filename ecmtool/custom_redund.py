import numpy as np
from scipy.linalg import LinAlgError
from scipy.optimize import linprog

from ecmtool.helpers import mp_print

try:
    from ecmtool._bglu_dense import BGLU
except (ImportError, EnvironmentError, OSError):
    from ecmtool.bglu_dense_uncompiled import BGLU
from ecmtool.intersect_directly_mpi import perturb_LP, normalize_columns, independent_rows, get_start_basis, \
    add_first_ray, get_more_basis_columns, setup_cycle_LP, cycle_check_with_output


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


def get_remove_metabolite_redund(R, reaction, verbose=True):
    column = R[:, reaction]
    for i in range(len(column)):
        if column[i] != 0:
            return i
    print("\tWarning: column with only zeros is part of cycle")
    return 0


def cancel_with_cycle_redund(R, met, cycle_ind, verbose=True, tol=1e-12):
    cancelling_reaction = R[:, cycle_ind]
    reactions_using_met = [i for i in range(R.shape[1]) if R[met, i] != 0 and i != cycle_ind]

    next_R = np.copy(R)
    to_be_dropped = [cycle_ind]

    for reac_ind in reactions_using_met:
        coeff_cycle = R[met, cycle_ind]
        coeff_reac = R[met, reac_ind]
        new_ray = R[:, reac_ind] - (coeff_reac / coeff_cycle) * R[:, cycle_ind]
        if sum(abs(new_ray)) > tol:
            next_R[:, reac_ind] = new_ray
        else:
            to_be_dropped.append(reac_ind)

    # Delete cycle ray that is now the only one that produces met, so has to be zero + rays that are full of zeros now
    next_R = np.delete(next_R, to_be_dropped, axis=1)

    return next_R


def remove_cycles_redund(R, tol=1e-12, verbose=True):
    """Detect whether there are cycles, by doing an LP. If LP is unbounded find a minimal cycle. Cancel one metabolite
    with the cycle."""
    cycle_inds = []
    A_ub, b_ub, A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows(normalize_columns(np.array(R, dtype='float'))))

    if verbose:
        mp_print('Constructing basis for LP')
    basis = get_more_basis_columns(np.asarray(A_eq, dtype='float'), [])
    b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, 1e-10)
    if verbose:
        mp_print('Starting linearity check using LP.')
    cycle_present, status, cycle_indices = cycle_check_with_output(c, np.asarray(A_eq, dtype='float'), x0, basis)

    if status != 0:
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12},
                      x0=x0)
        if res.status == 4:
            print("Numerical difficulties with revised simplex, trying interior point method instead")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

        cycle_present = True if np.max(res.x) > 90 else False
        if cycle_present:
            cycle_indices = np.where(res.x > 90)[0]
        if np.any(np.isnan(res.x)):
            raise Exception('Remove cycles did not work, because LP-solver had issues. Try to solve this.')

    # if the objective is unbounded, there is a cycle that sums to zero
    while cycle_present:
        # Find minimal cycle
        met = -1
        counter = 0
        while met < 0:
            cycle_ind = cycle_indices[counter]
            met = get_remove_metabolite_redund(R, cycle_ind)
            counter = counter + 1

        R = cancel_with_cycle_redund(R, met, cycle_ind)
        cycle_inds = np.append(cycle_inds, cycle_ind)

        # Do new LP to check if there is still a cycle present.
        A_ub, b_ub, A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows(normalize_columns(np.array(R, dtype='float'))))

        basis = get_more_basis_columns(np.asarray(A_eq, dtype='float'), [])
        b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, 1e-10)
        if verbose:
            mp_print('Starting linearity check in H_eq using LP.')
        cycle_present, status, cycle_indices = cycle_check_with_output(c, np.asarray(A_eq, dtype='float'), x0, basis)

        if status != 0:
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12},
                          x0=x0)
            if res.status == 4:
                print("Numerical difficulties with revised simplex, trying interior point method instead")
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

            cycle_present = True if np.max(res.x) > 90 else False
            if cycle_present:
                cycle_indices = np.where(res.x > 90)[0]
            if np.any(np.isnan(res.x)):
                raise Exception('Remove cycles did not work, because LP-solver had issues. Try to solve this.')

    return R, np.array(cycle_inds, dtype=int)


def pre_redund(matrix_indep_rows):
    """In this function, we try to filter out many redundant rays, without claiming that all redundant rays are
    filtered out. Running a redundancy-removal method is still needed after this."""
    start_basis = get_start_basis(matrix_indep_rows)
    start_basis_inv = np.linalg.inv(matrix_indep_rows[:, start_basis])

    filtered_inds = start_basis
    filtered_rays = matrix_indep_rows[:, start_basis]
    n_rays = matrix_indep_rows.shape[1]
    local_basis_inds = np.arange(len(start_basis))
    for i in range(n_rays):
        if i % 100 == 0:
            mp_print("Passed %d of %d (%f %%) through redundancy filter. Found %d redundant rays." %
                     (i, n_rays, i / n_rays * 100, i-len(np.where(filtered_inds<i)[0])))
        if i not in filtered_inds:
            new_ray = matrix_indep_rows[:, i][:, np.newaxis]
            filtered_rays_new = np.append(filtered_rays, new_ray, axis=1)
            basis = add_first_ray(filtered_rays_new, start_basis_inv, local_basis_inds, filtered_rays_new.shape[1] - 1)
            extreme = check_extreme(filtered_rays_new, filtered_rays_new.shape[1] - 1, basis)
            if extreme:
                filtered_rays = filtered_rays_new
                filtered_inds = np.append(filtered_inds, i)

    return filtered_inds


def drop_redundant_rays(ray_matrix, verbose=True, use_pre_filter=False):
    # Sometimes, use_pre_filter=True can speed up the calculations, but most of the times it doesn't

    # first find 'cycles': combinations of columns of matrix_indep_rows that add up to zero, and remove them
    if verbose:
        mp_print('Detecting linearities in H_ineq.')
    ray_matrix_wo_linearities, cycle_inds = remove_cycles_redund(ray_matrix)

    cycle_rays = ray_matrix[:, cycle_inds]

    if verbose:
        mp_print('Preparing redundancy test.')
    matrix_normalized = normalize_columns(ray_matrix_wo_linearities)
    matrix_indep_rows = independent_rows(matrix_normalized)

    if use_pre_filter:
        filtered_inds = pre_redund(matrix_indep_rows)
    else:
        filtered_inds = np.arange(matrix_indep_rows.shape[1])

    matrix_indep_rows = matrix_indep_rows[:, filtered_inds]

    # then find any column basis of R_indep
    start_basis = get_start_basis(matrix_indep_rows)
    start_basis_inv = np.linalg.inv(matrix_indep_rows[:, start_basis])

    number_rays = matrix_indep_rows.shape[1]
    init_number_rays = number_rays
    non_extreme_rays = []
    for i in range(number_rays):
        if i % 10 == 0:
            mp_print("Custom redund is on reduncancy test %d of %d (%f %%). Found %d redundant rays." %
                     (i, init_number_rays, i / init_number_rays * 100, len(non_extreme_rays)), PRINT_IF_RANK_NONZERO=True)

        col_ind = i - len(non_extreme_rays)
        basis = add_first_ray(matrix_indep_rows, start_basis_inv, start_basis, col_ind)
        extreme = check_extreme(matrix_indep_rows, col_ind, basis)

        if not extreme:
            non_extreme_rays.append(i)
            matrix_indep_rows = np.delete(matrix_indep_rows, col_ind, axis=1)
            if col_ind in start_basis:
                remaining_basis = np.delete(start_basis, np.where(start_basis == col_ind)[0])
                remaining_basis[remaining_basis > col_ind] -= 1
                start_basis = get_more_basis_columns(matrix_indep_rows, remaining_basis)
                start_basis_inv = np.linalg.inv(matrix_indep_rows[:, start_basis])
            else:
                start_basis[start_basis > col_ind] -= 1

            number_rays = number_rays - 1

    non_extreme_rays.sort()
    extreme_inds = np.delete(filtered_inds, non_extreme_rays)
    new_ray_matrix = ray_matrix_wo_linearities[:, extreme_inds]

    return new_ray_matrix, cycle_rays
