import multiprocessing as multi
from time import time

import numpy as np
from scipy.linalg import LinAlgError
from scipy.optimize import linprog

from ecmtool._bglu_dense import BGLU
from ecmtool.helpers import redund


def print_ecms_direct(R, metabolite_ids):
    obj_id = -1
    if "objective" in metabolite_ids:
        obj_id = metabolite_ids.index("objective")
    elif "objective_out" in metabolite_ids:
        obj_id = metabolite_ids.index("objective_out")

    print("\n--%d ECMs found by intersecting directly--\n" % R.shape[1])
    # for i in range(R.shape[1]):
    #     print("ECM #%d:" % i)
    #     div = 1
    #     if obj_id != -1 and R[obj_id][i] != 0:
    #         div = R[obj_id][i]
    #     for j in range(R.shape[0]):
    #         if R[j][i] != 0:
    #             print("%s: %f" % (metabolite_ids[j].replace("_in", "").replace("_out", ""), float(R[j][i]) / div))
    #     print("")


def get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape

    rank = np.linalg.matrix_rank(A[:, basis])
    new_basis = basis.copy()
    for i in range(n):
        if i in new_basis:
            continue
        prev_rank = rank
        prev_basis = new_basis
        new_basis = np.append(new_basis, i)
        rank = np.linalg.matrix_rank(A[:, new_basis])

        if rank == prev_rank:  # column added did not increase rank
            new_basis = prev_basis
        if rank == m:
            break

    return new_basis


def kkt_check(c, A, x, basis, tol=1e-8, threshold=1e-3, max_iter=1000, verbose=True):
    """
    Determine whether KKT conditions hold for x0.
    Take size 0 steps if available.
    """
    ab = np.arange(A.shape[0])
    a = np.arange(A.shape[1])

    maxupdate = 10
    B = BGLU(A, basis, maxupdate, False)
    for iteration in range(max_iter):
        bl = np.zeros(len(a), dtype=bool)
        bl[basis] = 1
        xb = x[basis]

        try:
            l = B.solve(c[basis], transposed=True)  # similar to v = linalg.solve(B.T, c[basis])
        except LinAlgError:
            return True, 1
        sn = c - l.dot(A)  # reduced cost
        sn = sn[~bl]

        if np.all(sn >= -tol):  # in this case x is an optimal solution
            # if verbose:
            #     print("Did %d steps in kkt_check, found True - smallest sn: %.8f" % (iteration - 1, min(sn)))
            return True, 0

        entering = a[~bl][np.argmin(sn)]
        u = B.solve(A[:, entering])

        i = u > tol  # if none of the u are positive, unbounded
        if not np.any(i):
            print("Warning: unbounded problem in KKT_check")
            # if verbose:
            #     print("Did %d steps in kkt_check2" % iteration - 1)
            return True, 1

        th = xb[i] / u[i]
        l = np.argmin(th)  # implicitly selects smallest subscript
        step_size = th[l]  # step size

        # Do pivot
        x[basis] = x[basis] - step_size * u
        x[entering] = step_size
        x[abs(x) < 10e-20] = 0
        B.update(ab[i][l], entering)  # modify basis
        basis = B.b

        if np.dot(c, x) < -threshold:  # found a better solution, so not adjacent
            # if verbose:
            #     print("Did %d steps in kkt_check, found False - c*x %.8f" % (iteration - 1, np.dot(c, x)))
            return False, 0

    print("Cycling?")
    return True, 1


def get_nonsingular_pair(A, basis, entering, leaving, basis_hashes):
    for enter in entering:
        for leave in leaving:
            original = basis[leave]
            basis[leave] = enter
            if np.linalg.matrix_rank(A[:, basis]) >= min(A[:, basis].shape):
                if hash(np.sort(basis).tostring()) in basis_hashes:
                    basis[leave] = original
                    continue
                return basis
            basis[leave] = original
    print("Did not find non-singular entering+leaving index...")
    basis[leaving[0]] = entering[0]
    return basis


def independent_rows(A):
    m, n = A.shape
    basis = np.asarray([], dtype='int')
    A_float = np.asarray(A, dtype='float')
    rank = np.linalg.matrix_rank(A_float)
    original_rank = rank

    if rank == m:
        return A

    rank = 0
    for i in range(m):
        prev_rank = rank
        prev_basis = basis
        basis = np.append(basis, i)
        rank = np.linalg.matrix_rank(A_float[basis])

        if rank == prev_rank:  # row added did not increase rank
            basis = prev_basis
        # else:
        #     print("added row, condition number is now: %f" % np.linalg.cond(A_float[basis]))
        #     if np.linalg.cond(A_float[basis]) > 1000:
        #         basis = prev_basis
        #         rank = np.linalg.matrix_rank(A_float[basis])
        #         print("Rejected column based on condition number...")

        if rank == original_rank:
            break

    if rank != original_rank:
        print("\t(rows) Rank deficiency! Got rank %d instead of %d" % (rank, m))
    return A[basis]


def cancel_with_cycle(R, met, cycle_ind, network, verbose=True, tol=1e-12):
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

    # delete all-zero row
    next_R = np.delete(next_R, met, 0)
    network.drop_metabolites([met])
    print("After removing this metabolite, we have %d metabolites and %d rays" %
          (next_R.shape[0], next_R.shape[1]))

    next_R = np.transpose(next_R)
    rows_before = next_R.shape[0]

    if verbose:
        print("\tDimensions before redund: %d %d" % (next_R.shape[0], next_R.shape[1]))
    start = time()
    next_R = redund(next_R)
    end = time()
    rows_removed_redund = rows_before - next_R.shape[0]
    if verbose:
        print("\tDimensions after redund: %d %d" % (next_R.shape[0], next_R.shape[1]))
        print("\t\tRows removed by redund: %d" % (rows_before - next_R.shape[0]))
        print("\tRedund took %f seconds" % (end - start))

    next_R = np.transpose(next_R)

    return next_R


def eliminate_metabolite(R, met, network, calculate_adjacency=True, tol=1e-12, verbose=True,
                         lps_per_job=1):
    # determine +/0/-
    plus = []
    zero = []
    minus = []
    for reaction in range(R.shape[1]):
        result = R[met, reaction]
        if abs(result) <= tol:
            zero.append(reaction)
        elif result > tol:
            plus.append(reaction)
        elif result < -tol:
            minus.append(reaction)
        else:
            zero.append(reaction)
    if verbose:
        print("\tNumber of +: %d" % len(plus))
        print("\tNumber of -: %d" % len(minus))
        print("\tNumber of LP to do: %d" % (len(plus) * len(minus)))

    # start next matrix with zero rows
    next_matrix = []
    for z in zero:
        col = R[:, z]
        next_matrix.append(col)

    if calculate_adjacency:
        adj = geometric_ray_adjacency(R, plus=plus, minus=minus, verbose=verbose,
                                      remove_cycles=True, lps_per_job=lps_per_job)

    # combine + and - if adjacent
    nr_adjacent = 0
    for p in plus:
        for m in minus:
            if not calculate_adjacency or adj[p, m] == 1:
                nr_adjacent += 1
                rp = R[met, p]
                rm = R[met, m]
                new_row = rp * R[:, m] - rm * R[:, p]
                if sum(abs(new_row)) > tol:
                    next_matrix.append(new_row)

    if verbose:
        if len(plus) * len(minus) > 0:
            print("Of %d candidates, %d were adjacent (%.2f percent)" % (
                len(plus) * len(minus), nr_adjacent, 100 * nr_adjacent / (len(plus) * len(minus))))
        else:
            print("Of %d candidates, %d were adjacent (0 percent)" % (len(plus) * len(minus), nr_adjacent))

    next_matrix = np.asarray(next_matrix)

    rows_removed_redund = 0
    # redund in case we have too many rows
    if not calculate_adjacency:
        rows_before = next_matrix.shape[0]

        if verbose:
            print("\tDimensions before redund: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))
        start = time()
        next_matrix = redund(next_matrix)
        end = time()
        rows_removed_redund = rows_before - next_matrix.shape[0]
        if verbose:
            print("\tDimensions after redund: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))
            print("\t\tRows removed by redund: %d" % (rows_before - next_matrix.shape[0]))
            print("\tRedund took %f seconds" % (end - start))

    next_matrix = np.transpose(next_matrix)

    # delete all-zero row
    next_matrix = np.delete(next_matrix, met, 0)
    network.drop_metabolites([met])
    print("After removing this metabolite, we have %d metabolites and %d rays" %
          (next_matrix.shape[0], next_matrix.shape[1]))

    return next_matrix, rows_removed_redund


def get_remove_metabolite(R, network, reaction, verbose=True):
    column = R[:, reaction]
    for i in range(len(column)):
        if not network.metabolites[i].is_external:
            if column[i] != 0:
                return i
    print("\tWarning: reaction to augment has only external metabolites")
    return 0


def remove_cycles(R, network, tol=1e-12, verbose=True):
    """Detect whether there are cycles, by doing an LP. If LP is unbounded find a minimal cycle. Cancel one metabolite
    with the cycle."""
    deleted = []
    A_ub, b_ub, A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows(normalize_columns(np.array(R, dtype='float'))))

    # TODO: Erik, can you add your LP solver here? These ones do not work.
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12},
                  x0=x0)
    if res.status == 4:
        print("Numerical difficulties with revised simplex, trying interior point method instead")
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

    # if the objective is unbounded, there is a cycle that sums to zero
    while np.max(res.x) > 90:  # It is unbounded
        # Find minimal cycle
        cycle_indices = np.where(res.x > 90)[0]
        met = -1
        counter = 0
        while met < 0:
            cycle_ind = cycle_indices[counter]
            met = get_remove_metabolite(R, network, cycle_ind)
            counter = counter + 1
            if counter > len(cycle_indices):
                print('No internal metabolite was found that was part of the cycle. This might cause problems.')

        deleted.append(met)
        if verbose:
            print("Found an unbounded LP, cancelling metabolite %d (%s) with reaction %d" % (
                met, network.metabolites[met].id, cycle_ind))

        R = cancel_with_cycle(R, met, cycle_ind, network)

        A_ub, b_ub, A_eq, b_eq, c, x0 = setup_cycle_LP(independent_rows(normalize_columns(np.array(R, dtype='float'))))

        # TODO: Erik, can you add your LP solver here? These ones do not work.
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12},
                      x0=x0)
        if res.status == 4:
            print("Numerical difficulties with revised simplex, trying interior point method instead")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

    internal_metabolite_indices = [i for i, metab in enumerate(network.metabolites) if not metab.is_external]
    removable_metabolites = []
    for metabolite_index in internal_metabolite_indices:
        nonzero_count = np.count_nonzero(network.N[metabolite_index, :])
        if nonzero_count == 0:
            removable_metabolites.append(metabolite_index)

    network.drop_metabolites(removable_metabolites)

    return R, deleted


def normalize_columns(R):
    result = R.copy()
    for i in range(result.shape[1]):
        result[:, i] /= np.linalg.norm(np.array(R[:, i], dtype='float'))
    return result


def smallest_positive(arr):
    a = np.where(np.isfinite(arr), arr, -1)
    return min(np.where(a < 0, max(a) * 2, a)), np.argmin(np.where(a < 0, max(a) * 2, a))


def generate_BFS(R, i, j, eps):
    ray1 = np.array(np.concatenate((R[:, i], -R[:, i])), dtype='float')
    ray2 = np.array(np.concatenate((R[:, j], -R[:, j])), dtype='float')
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha, k = smallest_positive(eps / ray1)
        beta = ray2[k] / ray1[k]
        arr = (eps - alpha * ray1) / (ray2 - beta * ray1)
        arr[k] = -1  # ignore place k, because it should always be divide by 0
        delta2, _ = smallest_positive(arr)
        delta1 = -beta * delta2
    sbar = eps - (alpha + delta1) * ray1 - delta2 * ray2
    l = np.zeros(R.shape[1])
    l[i] = 0.5 + alpha + delta1
    l[j] = 0.5 + delta2

    res = np.concatenate((l, sbar))
    # round to 0 when a rounding error made it non-zero
    res = np.where(abs(res) < 1e-20, 0, res)

    if len(res[res != 0]) != R.shape[0] * 2:
        print("problem in generate_BFS")

    return res


def setup_LP_perturbed(R, i, j, epsilon):
    m, n = R.shape

    A_ub = -np.identity(n + 2 * m)
    b_ub = np.zeros(n + 2 * m)
    A_eq = np.concatenate((np.concatenate((R, -R)), np.identity(2 * m)), axis=1)
    ray1 = R[:, i]
    ray2 = R[:, j]
    tar = 0.5 * ray1 + 0.5 * ray2
    eps_vector = np.array([epsilon] * (2 * m)) + np.random.uniform(-epsilon / 2, epsilon / 2, 2 * m)
    b_eq = np.concatenate((tar, -tar)) + eps_vector
    x0 = generate_BFS(R, i, j, eps_vector)
    c = np.concatenate((-np.ones(n), np.zeros(2 * m)))
    c[i] = 0
    c[j] = 0

    return A_ub, b_ub, A_eq, b_eq, c, x0


def setup_cycle_detection_LP(R_indep, cycle_ind):
    number_rays = R_indep.shape[1]

    A_ub = -np.identity(number_rays)
    b_ub = np.zeros(number_rays)
    A_eq = R_indep
    b_eq = np.zeros(A_eq.shape[0])

    # Add that the cycle_ind should be used
    extra_row = np.zeros((1, A_eq.shape[1]))
    extra_row[0, cycle_ind] = 1
    A_eq2 = np.concatenate((A_eq, extra_row), axis=0)
    b_eq2 = np.zeros(len(b_eq) + 1)
    b_eq2[-1] = 1

    c = np.ones(number_rays)

    return A_ub, b_ub, A_eq2, b_eq2, c


def setup_cycle_LP(R_indep):
    number_rays = R_indep.shape[1]

    A_ub = -np.identity(number_rays)
    b_ub = np.zeros(number_rays)
    A_eq = R_indep
    b_eq = np.zeros(A_eq.shape[0])
    c = -np.ones(number_rays)

    # Add upper bound of 100
    A_ub2 = np.concatenate((A_ub, np.identity(number_rays)))
    b_ub2 = np.concatenate((b_ub, [100] * number_rays))

    x0 = np.zeros(number_rays)

    return A_ub2, b_ub2, A_eq, b_eq, c, x0


def setup_LP(R_indep, i, j):
    number_rays = R_indep.shape[1]

    A_ub = -np.identity(number_rays)
    b_ub = np.zeros(number_rays)
    A_eq = R_indep
    ray1 = R_indep[:, i]
    ray2 = R_indep[:, j]
    b_eq = 0.5 * ray1 + 0.5 * ray2
    c = -np.ones(number_rays)
    c[i] = 0
    c[j] = 0
    x0 = np.zeros(number_rays)
    x0[i] = 0.5
    x0[j] = 0.5

    return A_ub, b_ub, A_eq, b_eq, c, x0


def perturb_LP(b_eq, x0, A_eq, basis, epsilon):
    eps = np.random.uniform(epsilon / 2, epsilon, len(basis))
    b_eq = b_eq + np.dot(A_eq[:, basis], eps)
    x0[basis] += eps

    return b_eq, x0


def determine_adjacency(R, i, j, start_basis, tol=1e-10):
    A_ub, b_ub, A_eq, b_eq, c, x0 = setup_LP(R, i, j)
    # ext_basis = get_more_basis_columns(np.asarray(A_eq, dtype='float'), [i, j])
    ext_basis = add_rays(np.asarray(A_eq, dtype='float'), start_basis, i, j)
    b_eq, x0 = perturb_LP(b_eq, x0, A_eq, ext_basis, 1e-10)
    KKT, status = kkt_check(c, np.asarray(A_eq, dtype='float'), x0, ext_basis)

    if status == 0:
        return 1 if KKT else 0
    else:
        print("\t\t\tKKT had non-zero exit status...")
        # input("Waiting...")
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex',
                      options={'tol': 1e-12, 'maxiter': 500})

        if res.status == 1:
            print("Iteration limit %d reached, trying Blands pivot rule" % (res.nit))
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex',
                          options={'tol': 1e-12, 'pivot': "Bland", 'maxiter': 20000})

        if res.status == 4:
            print("Numerical difficulties with revised simplex, trying interior point method instead")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point',
                          options={'tol': 1e-12})

        if res.status != 0:
            print("Status %d" % res.status)
            # input("Waiting...")

        print("res.fun: %.2e res.nit: %d" % (abs(res.fun), res.nit))
        if res.status != 0 or abs(res.fun) < tol:
            return 1

        return 0


def multiple_adjacencies(R, pairs, start_basis):
    return [(p, determine_adjacency(R, p[0], p[1], start_basis)) for p in pairs]


def unpack_results(results, number_rays, adjacency=None):
    if adjacency is None:
        adjacency = np.zeros(shape=(number_rays, number_rays))
    for result in results:
        for line in result:
            i = line[0][0]
            j = line[0][1]
            adjacency[i][j] = line[1]
    return adjacency


def add_second_ray(A, basis_p, p, m):
    res = np.copy(basis_p)
    if m in basis_p:
        return res

    # Faster, but doesnt work (get singular matrix)
    # x = np.linalg.solve(B_plus_inv, A[:, m])
    # x[np.where(basis_p == p)[0][0]] = 0 # dont select p for replacement in basis
    # res[np.argmax(x)] = m
    # return res

    for i in range(len(basis_p)):
        if basis_p[i] == p:
            continue
        res[i] = m
        rank = np.linalg.matrix_rank(A[:, res])
        if rank == A.shape[0]:
            return res
        res[i] = basis_p[i]


def add_first_ray(A, start_basis, p):
    res = np.copy(start_basis)
    if p in start_basis:
        return res

    # x = np.linalg.solve(B_inv, A[:, p])
    # res[np.argmax(x)] = p
    # return res

    for i in range(len(start_basis)):
        res[i] = p
        rank = np.linalg.matrix_rank(A[:, res])
        if rank == A.shape[0]:
            return res
        res[i] = start_basis[i]


def add_rays(A, start_basis, p, m):
    basis = add_first_ray(A, start_basis, p)
    basis = add_second_ray(A, basis, p, m)
    return basis


def get_start_basis(A):
    # return any column basis of A
    m, n = A.shape

    rank = 0
    new_basis = np.array([], dtype='int')
    for i in range(n):
        if i in new_basis:
            continue
        prev_rank = rank
        prev_basis = new_basis
        new_basis = np.append(new_basis, i)
        rank = np.linalg.matrix_rank(A[:, new_basis])

        if rank == prev_rank:  # column added did not increase rank
            new_basis = prev_basis
        # else:
        #     print("added column, condition number is now: %f" % np.linalg.cond(A[:, new_basis]))
        #     if np.linalg.cond(A[:, new_basis]) > 1000:
        #         new_basis = prev_basis
        #         rank = np.linalg.matrix_rank(A[:, new_basis])
        #         print("Rejected column based on condition number...")

        if rank == m:
            break

    if rank != m:
        print("\tRank deficiency! Got rank %d instead of %d" % (rank, m))
    return new_basis


def get_bases(A, plus, minus):
    m = A.shape[0]
    bases = np.zeros((len(plus), len(minus), m))

    start_basis = get_start_basis(A)
    # B_inv = np.linalg.inv(A[:, start_basis])

    for p in plus:
        basis_p = add_first_ray(A, start_basis, p)
        # B_plus_inv = np.linalg.inv(A[:, basis_p])
        for m in minus:
            basis_pm = add_second_ray(A, basis_p, p, m)
            pass

    return bases


def geometric_ray_adjacency(R, plus=[-1], minus=[-1], tol=1e-3, verbose=True, remove_cycles=True,
                            lps_per_job=1):
    """
    Returns r by r adjacency matrix of rays, given
    ray matrix R. Diagonal is 0, not 1.
    Calculated using LP adjacency test
    :param R: ray matrix (columns are generating rays)
        plus: indices of 'plus' columns
        minus: indices of 'minus' columns
        if plus/minus are not provided, find all adjacencies; otherwise only between each + and - pair
    :return: r by r adjacency matrix
    """
    start = time()

    R_normalized = normalize_columns(np.array(R, dtype='float'))
    R_indep = independent_rows(R_normalized)

    # set default plus and minus
    if (len(plus) > 0 and plus[0] == -1):
        plus = [x for x in range(R_indep.shape[1])]
    if (len(minus) > 0 and minus[0] == -1):
        minus = [x for x in range(R_indep.shape[1])]

    number_rays = R_indep.shape[1]

    # bases = get_bases(R_indep, plus, minus)

    cpu_count = multi.cpu_count()
    print("Using %d cores" % cpu_count)

    pairs = np.array([(i, j) for i in plus for j in minus])
    split_indices = [i for i in range(1, len(pairs)) if i % lps_per_job == 0]
    split_pairs = np.split(pairs, split_indices)
    q = 1000
    split_indices = [i for i in range(1, len(split_pairs)) if i % (q / lps_per_job) == 0]
    report_unit = np.split(split_pairs, split_indices)
    adjacency = None
    LPs_done = 0
    start_time = time()
    start_basis = get_start_basis(R_indep)
    with multi.Pool(cpu_count) as pool:
        for unit in report_unit:
            result = pool.starmap(multiple_adjacencies, [(R_indep, pairs, start_basis) for pairs in unit])

            adjacency = unpack_results(result, number_rays, adjacency)
            LPs_done += int(q / lps_per_job) * lps_per_job
            if len(pairs) != 0:
                duration = time() - start_time
                fraction_done = min(LPs_done, len(pairs)) / len(pairs)
                est_total_time = duration / fraction_done
                time_remaining = est_total_time - duration
                print("Did {} of the LPs ({:.2f} percent). Estimated time remaining for this step: {:.2f}s".format(
                    min(LPs_done, len(pairs)), 100 * fraction_done, time_remaining))

    end = time()
    print("Did LPs in %f seconds" % (end - start))
    return adjacency


def reduce_column_norms(matrix):
    for i in range(matrix.shape[1]):
        norm = np.linalg.norm(np.array(matrix[:, i], dtype='float'))
        if norm > 2:
            matrix[:, i] /= int(np.floor(norm))
    return matrix


def unsplit_metabolites(R, network):
    metabolite_ids = [network.metabolites[i].id for i in network.external_metabolite_indices()]
    res = []
    ids = []

    processed = {}
    for i in range(R.shape[0]):
        metabolite = metabolite_ids[i].replace("_in", "").replace("_out", "")
        if metabolite in processed:
            row = processed[metabolite]
            res[row] += R[i, :]
        else:
            res.append(R[i, :].tolist())
            processed[metabolite] = len(res) - 1
            ids.append(metabolite)

    # remove all-zero rays
    res = np.asarray(res)
    res = res[:, [sum(abs(res)) != 0][0]]

    return res, ids


def in_cone(R, tar):
    number_rays = R.shape[1]

    A_ub = -np.identity(number_rays)
    b_ub = np.zeros(number_rays)
    A_eq = R
    b_eq = tar
    c = -np.ones(number_rays)

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12})

    return A_ub, b_ub, A_eq, b_eq, c


def intersect_directly(R, internal_metabolites, network, verbose=True, tol=1e-12, lps_per_job=1):
    # rows are rays
    deleted = np.array([])
    it = 1
    internal = list(internal_metabolites)
    internal.sort()
    rows_removed_redund = 0

    while len(internal) > 0:
        i = internal[np.argmin(
            [np.sum(R[j - len(deleted[deleted < j]), :] > 0) * np.sum(R[j - len(deleted[deleted < j]), :] < 0) for j in
             internal])]
        to_remove = i - len(deleted[deleted < i])
        if verbose:
            print("\n\nIteration %d (internal metabolite = %d: %s) of %d" % (
                it, to_remove, [m.id for m in network.metabolites][to_remove], len(internal_metabolites)))
            print("Possible LP amounts for this step:\n" + ", ".join(np.sort(
                [np.sum(R[j - len(deleted[deleted < j]), :] > 0) * np.sum(R[j - len(deleted[deleted < j]), :] < 0) for j
                 in internal]).astype(str)))
            print("Total: %d" % sum(
                [np.sum(R[j - len(deleted[deleted < j]), :] > 0) * np.sum(R[j - len(deleted[deleted < j]), :] < 0) for j
                 in internal]))
            it += 1

        # input("waiting")
        # TODO make iteration_without_lps
        # if np.sum(R[i - len(deleted[deleted < i]), :] > 0) * np.sum(R[i - len(deleted[deleted < i]), :] < 0) == 0:
        #     R, removed = iteration_without_lps(R, to_remove, network)

        R, removed = eliminate_metabolite(R, to_remove, network, calculate_adjacency=True, lps_per_job=lps_per_job)
        rows_removed_redund += removed
        deleted = np.append(deleted, i)
        internal.remove(i)

    # remove artificial rays introduced by splitting metabolites
    R, ids = unsplit_metabolites(R, network)

    if verbose:
        print("\n\tRows removed by redund overall: %d\n" % rows_removed_redund)
        if rows_removed_redund != 0:
            pass
            # input("Waiting...")

    return R, ids
