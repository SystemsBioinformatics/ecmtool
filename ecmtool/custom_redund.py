import numpy as np
from scipy.linalg import LinAlgError
from scipy.optimize import linprog

from ecmtool import mpi_wrapper
from ecmtool.helpers import mp_print, to_fractions, normalize_columns, normalize_columns_fraction

try:
    from ecmtool._bglu_dense import BGLU
except (ImportError, EnvironmentError, OSError):
    from ecmtool.bglu_dense_uncompiled import BGLU
from ecmtool.intersect_directly_mpi import perturb_LP, independent_rows_qr, get_start_basis, \
    add_first_ray, get_more_basis_columns, setup_cycle_LP, cycle_check_with_output, get_basis_columns_qr


def unique(matrix):
    unique_set = list({tuple(row) for row in matrix if np.count_nonzero(row) > 0})
    return np.vstack(unique_set) if len(unique_set) else to_fractions(np.ndarray(shape=(0, matrix.shape[1])))


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
            return True, 0

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
        if iteration % max_iter == 0:
            mp_print("Cycling? Starting again with new perturbation.")
            return True, 2

    return True, 1


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

    counter_seeds = 1
    while status == 2:
        tol = tol + 1e-10
        b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, tol, seed=42 + counter_seeds)
        KKT, status = kkt_check_redund(c, A_eq, x0, basis, i)
        counter_seeds = counter_seeds + 1
        if counter_seeds % 20 == 0:
            mp_print(
                'Warning: Adjacency check keeps cycling, even with different perturbations. Reporting rays as adjacent.',
                PRINT_IF_RANK_NONZERO=True)
            status = 0
            KKT = True

    if status == 0:
        return KKT
    else:
        mp_print('LinAlgError in an adjacency test. Check if this happens more often.', PRINT_IF_RANK_NONZERO=True)
        mp_print('Now assuming that rays are adjacent.', PRINT_IF_RANK_NONZERO=True)
        return True


def get_remove_metabolite_redund(R, reaction, verbose=True):
    column = R[:, reaction]
    metab_occupancy = [(ind, np.count_nonzero(R[ind, :])) for ind in range(len(column)) if column[ind] != 0]
    if not len(metab_occupancy):
        print("\tWarning: column with only zeros is part of cycle")
        return -1

    # Choose minimally involved metabolite
    return min(metab_occupancy, key=lambda x: x[1])[0]


def cancel_with_cycle_redund(R, met, cycle_ind, verbose=True, tol=1e-12):
    cancelling_reaction = R[:, cycle_ind]
    reactions_using_met = [i for i in range(R.shape[1]) if R[met, i] != 0 and i != cycle_ind]

    cycle_ray = R[:, cycle_ind]
    # next_R = np.copy(R)
    to_be_dropped = [cycle_ind]

    n_reacs = len(reactions_using_met)
    for enum_ind, reac_ind in enumerate(reactions_using_met):
        if verbose:
            if enum_ind % 10000 == 0:
                mp_print("Removed cycle metab from %d of %d reactions (%f %%)" %
                         (enum_ind, n_reacs, enum_ind / n_reacs * 100))
        coeff_cycle = cycle_ray[met]
        coeff_reac = R[met, reac_ind]
        new_ray = R[:, reac_ind] - (coeff_reac / coeff_cycle) * cycle_ray
        if sum(abs(new_ray)) > tol:
            R[:, reac_ind] = new_ray
        else:
            to_be_dropped.append(reac_ind)

    # Delete cycle ray that is now the only one that produces met, so has to be zero + rays that are full of zeros now
    R = np.delete(R, to_be_dropped, axis=1)

    return R


def remove_cycles_redund(R, tol=1e-12, verbose=True):
    """Detect whether there are cycles, by doing an LP. If LP is unbounded find a minimal cycle. Cancel one metabolite
    with the cycle."""
    zero_rays = np.where(np.count_nonzero(R, axis=0) == 0)[0]
    for ray_ind in zero_rays:
        R = np.delete(R, ray_ind, axis=1)
    cycle_rays = np.zeros((R.shape[0], 0))
    A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows_qr(normalize_columns(np.array(R, dtype='float'))), only_eq=True)

    if verbose:
        mp_print('Constructing basis for LP')
    basis = get_basis_columns_qr(np.asarray(A_eq, dtype='float'))
    b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, 1e-10)
    if verbose:
        mp_print('Starting linearity check using LP.')
    cycle_present, status, cycle_indices = cycle_check_with_output(c, np.asarray(A_eq, dtype='float'), x0, basis)

    if status != 0:
        print("Cycle check failed, trying normal LP")
        A_ub, b_ub, A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows_qr(normalize_columns(np.array(R, dtype='float'))))
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

        cycle_rays = np.append(cycle_rays, R[:, cycle_ind][:, np.newaxis], axis=1)
        R = cancel_with_cycle_redund(R, met, cycle_ind)

        # Do new LP to check if there is still a cycle present.
        A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows_qr(normalize_columns(np.array(R, dtype='float'))), only_eq=True)

        basis = get_basis_columns_qr(np.asarray(A_eq, dtype='float'))
        b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, 1e-10)
        if verbose:
            mp_print('Starting linearity check in H_ineq using LP.')
        cycle_present, status, cycle_indices = cycle_check_with_output(c, np.asarray(A_eq, dtype='float'), x0, basis)

        if status != 0:
            print("Cycle check failed, trying normal LP")
            A_ub, b_ub, A_eq, b_eq, c, x0 = setup_cycle_LP(
                independent_rows_qr(normalize_columns(np.array(R, dtype='float'))))
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

    return R, cycle_rays


def pre_redund(matrix_indep_rows):
    """In this function, we try to filter out many redundant rays, without claiming that all redundant rays are
    filtered out. Running a redundancy-removal method is still needed after this."""
    start_basis = get_basis_columns_qr(matrix_indep_rows)
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


def drop_redundant_rays(ray_matrix, verbose=True, use_pre_filter=False, rays_are_unique=True, linearities=False, normalised=True):
    """

    :param ray_matrix:
    :param verbose:
    :param use_pre_filter: Sometimes, use_pre_filter=True can speed up the calculations, but mostly it doesn't
    :param rays_are_unique: Boolean that states whether rays given as input are already unique
    :param linearities: Boolean indicating if linearities are still present
    :param normalised: Boolean indicating if ray_matrix columns are already normalised
    :return:
    """
    extreme_inds = []
    non_extreme_inds = []
    if not rays_are_unique:
        # First make sure that no duplicate rays are in the matrix
        ray_matrix = np.transpose(unique(np.transpose(normalize_columns_fraction(ray_matrix))))

    # Find 'cycles': combinations of columns of matrix_indep_rows that add up to zero, and remove them
    if linearities:
        if verbose:
            mp_print('Detecting linearities in H_ineq.')
        ray_matrix, cycle_rays = remove_cycles_redund(ray_matrix)
    else:
        cycle_rays = np.zeros((ray_matrix.shape[0], 0))

    if not normalised:
        if verbose:
            mp_print('Normalizing columns.')
        matrix_normalized = normalize_columns(ray_matrix)
    else:
        matrix_normalized = np.array(ray_matrix, dtype='float')

    if verbose:
        mp_print('Selecting independent rows.')
    matrix_indep_rows = independent_rows_qr(matrix_normalized)

    if use_pre_filter:
        filtered_inds = pre_redund(matrix_indep_rows)
    else:
        filtered_inds = np.arange(matrix_indep_rows.shape[1])

    mpi_size = mpi_wrapper.get_process_size()
    mpi_rank = mpi_wrapper.get_process_rank()

    matrix_indep_rows = matrix_indep_rows[:, filtered_inds]

    # then find any column basis of R_indep
    start_basis = get_basis_columns_qr(matrix_indep_rows)
    start_basis_inv = np.linalg.inv(matrix_indep_rows[:, start_basis])

    number_rays = matrix_indep_rows.shape[1]
    for i in range(number_rays):
        if i % mpi_size == mpi_rank:
            if i % (10 * mpi_size) == mpi_rank:
                mp_print("Process %d is on redundancy test %d of %d (%f %%). Found %d redundant rays." %
                         (mpi_rank, i, number_rays, i / number_rays * 100, len(non_extreme_inds)), PRINT_IF_RANK_NONZERO=True)
            basis = add_first_ray(matrix_indep_rows, start_basis_inv, start_basis, i)
            extreme = check_extreme(matrix_indep_rows, i, basis)
            if extreme:
                extreme_inds.append(filtered_inds[i])
            else:
                non_extreme_inds.append(filtered_inds[i])

    # MPI communication step
    extreme_sets = mpi_wrapper.world_allgather(extreme_inds)
    for i in range(mpi_size):
        if i != mpi_rank:
            extreme_inds.extend(extreme_sets[i])
    extreme_inds.sort()

    return extreme_inds, cycle_rays
