import numpy as np
from time import time
from scipy.optimize import linprog
from scipy.linalg import LinAlgError
import multiprocessing as multi

from ecmtool.helpers import redund
from ecmtool._bglu_dense import BGLU

def fake_ecm(reaction, metabolite_ids, tol=1e-12):
    s = ""
    for i, c in enumerate(np.asarray(reaction, dtype='int')):
        if abs(reaction[i]) > tol:
            if s == "":
                s = metabolite_ids[i].replace("_in", "").replace("_out", "")
            elif s != metabolite_ids[i].replace("_in", "").replace("_out", ""):
                return False
    return True


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

    # if (len(basis) > 0 and np.linalg.matrix_rank(A[:,basis]) < len(basis)):
    #    raise Exception("Basis has dependent columns")

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
            if verbose:
                print("Did %d steps in kkt_check, found True - smallest sn: %.8f" % (iteration - 1, min(sn)))
            return True, 0

        entering = a[~bl][np.argmin(sn)]
        u = B.solve(A[:, entering])

        i = u > tol  # if none of the u are positive, unbounded
        if not np.any(i):
            print("Warning: unbounded problem in KKT_check")
            if verbose:
                print("Did %d steps in kkt_check2" % iteration - 1)
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
            if verbose:
                print("Did %d steps in kkt_check, found False - c*x %.8f" % (iteration - 1, np.dot(c, x)))
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
        if rank == original_rank:
            break

    return A[basis]


def eliminate_metabolite(R, met, network, calculate_adjacency=True, tol=1e-12, perturbed=False, verbose=True):
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
        adj = geometric_ray_adjacency(R, plus=plus, minus=minus, perturbed=perturbed, verbose=verbose,
                                      remove_cycles=True)

        # combine + and - if adjacent
    nr_adjacent = 0
    for i, p in enumerate(plus):
        for j, m in enumerate(minus):
            if not calculate_adjacency or adj[i, j] == 1:
                nr_adjacent += 1
                rp = R[met, p]
                rm = R[met, m]
                new_row = rp * R[:, m] - rm * R[:, p]
                if sum(abs(new_row)) > tol:
                    next_matrix.append(new_row)

    if verbose:
        if len(plus) * len(minus) > 0:
            print("Of %d candidates, %d were adjacent (%f percent)" % (
                len(plus) * len(minus), nr_adjacent, 100 * nr_adjacent / (len(plus) * len(minus))))
        else:
            print("Of %d candidates, %d were adjacent (0 percent)" % (len(plus) * len(minus), nr_adjacent))

    next_matrix = np.asarray(next_matrix)



    # redund in case we have too many rows
    rows_before = next_matrix.shape[0]

    if verbose:
        print("\tDimensions before redund: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))
    start = time()
    # next_matrix = redund(next_matrix)
    end = time()
    rows_removed_redund = rows_before - next_matrix.shape[0]
    if verbose:
        print("\tDimensions after redund: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))
        print("\t\tRows removed by redund: %d" % (rows_before - next_matrix.shape[0]))
        print("\tRedund took %f seconds" % (end - start))
        # if rows_before - next_matrix.shape[0] != 0:
        #    input("Waiting...")

    next_matrix = np.transpose(next_matrix)

    # delete all-zero row
    next_matrix = np.delete(next_matrix, met, 0)
    network.drop_metabolites([met])
    print("\tDimensions after deleting row: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))

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
    deleted = []
    for k in range(2):
        number_rays = independent_rows(normalize_columns(np.array(R, dtype='float'))).shape[1]
        i = 0 + 2 * k
        j = 1 + 2 * k
        if j > R.shape[1] - 1:
            return R, deleted
        A_ub, b_ub, A_eq, b_eq, c, x0 = setup_LP(independent_rows(normalize_columns(np.array(R, dtype='float'))), i, j)

        if sum(abs(b_eq)) < tol:
            augment_reaction = i;
            met = get_remove_metabolite(R, network, augment_reaction)
            if verbose:
                print("Found an unbounded LP, augmenting reaction %d through metabolite %d" % (augment_reaction, met))
            R, _ = eliminate_metabolite(R, met, network, calculate_adjacency=False)

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12},
                      x0=x0)
        if res.status == 4:
            print("Numerical difficulties with revised simplex, trying interior point method instead")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

        # if the objective is unbounded, there is a cycle that sums to zero
        while res.status == 3:  # status 3 is unbounded
            A_ub2 = np.concatenate((A_ub, np.identity(number_rays)))
            b_ub2 = np.concatenate((b_ub, [100] * number_rays))

            res2 = linprog(c, A_ub2, b_ub2, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12}, x0=x0)
            if res2.status == 4:
                print("Numerical difficulties with revised simplex, trying interior point method instead")
                res2 = linprog(c, A_ub2, b_ub2, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

            if abs(res2.fun) < tol:  # res is 'unbounded' but res2 has optimum 0
                break

            augment_reaction = [i for i, val in enumerate(res2.x) if val > 90][0]
            met = get_remove_metabolite(R, network, augment_reaction)
            deleted.append(met)
            if verbose:
                print("Found an unbounded LP, augmenting reaction %d through metabolite %d (%s)" % (
                augment_reaction, met, network.metabolites[met].id))

            R, _ = eliminate_metabolite(R, met, network, calculate_adjacency=False)
            number_rays = independent_rows(normalize_columns(np.array(R, dtype='float'))).shape[1]
            i = 0 + 2 * k
            j = 1 + 2 * k
            A_ub, b_ub, A_eq, b_eq, c, x0 = setup_LP(independent_rows(normalize_columns(np.array(R, dtype='float'))), i,
                                                     j)

            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12}, x0=x0)
            if res.status == 4:
                print("Numerical difficulties with revised simplex, trying interior point method instead")
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

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


def determine_adjacency(R, i, j, perturbed, tol=1e-10):
    if perturbed:
        A_ub, b_ub, A_eq, b_eq, c, x0 = setup_LP_perturbed(R, i, j, 1e-10)
    else:
        A_ub, b_ub, A_eq, b_eq, c, x0 = setup_LP(R, i, j)

    # KKT
    disable_lp = True
    if perturbed:
        ext_basis = np.nonzero(x0)[0]
    else:
        ext_basis = get_more_basis_columns(np.asarray(A_eq, dtype='float'), [i, j])
    KKT, status = kkt_check(c, np.asarray(A_eq, dtype='float'), x0, ext_basis)
    # DEBUG
    # status = 0

    if status == 0:
        return 1 if KKT else 0

    print("\t\t\tKKT had non-zero exit status...")
    # input("Waiting...")
    disable_lp = False

    if not disable_lp:
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


def geometric_ray_adjacency(R, plus=[-1], minus=[-1], tol=1e-3, perturbed=False, verbose=True, remove_cycles=True):
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

    # with normalization
    R_normalized = normalize_columns(np.array(R, dtype='float'))
    R_indep = independent_rows(R_normalized)
    # without normalization
    # R_indep = independent_rows(R)

    LPs_done = 0
    # set default plus and minus
    if (len(plus) > 0 and plus[0] == -1):
        plus = [x for x in range(R_indep.shape[1])]
    if (len(minus) > 0 and minus[0] == -1):
        minus = [x for x in range(R_indep.shape[1])]

    number_rays = R_indep.shape[1]
    adjacency = np.zeros(shape=(number_rays, number_rays))

    disable_lp = not remove_cycles
    total = len(plus) * len(minus)

    print("\n\tLargest non-LP ray: %.2f" % max(
        [np.linalg.norm(np.array(R[:, i], dtype='float')) for i in range(R.shape[1])]))
    print("\tMax/min: %.3f" % max(
        [abs(abs(np.array(R[:, i], dtype='float')).max() / np.min(
            abs(np.array(R[:, i], dtype='float'))[np.nonzero(R[:, i])])) for i in range(R.shape[1])]))
    print("\tLargest LP ray: %.2f" % max(
        [np.linalg.norm(np.array(R_indep[:, i], dtype='float')) for i in range(R_indep.shape[1])]))

    with multi.Pool(multi.cpu_count()) as pool:
        adjacency_as_list = pool.starmap(determine_adjacency, [(R, i, j, perturbed) for i in plus for j in minus])
        adjacency = np.array(adjacency_as_list)
        adjacency = adjacency.reshape((len(plus), len(minus)))

    end = time()
    print("Did %d LPs in %f seconds" % (LPs_done, end - start))
    return adjacency


def reduce_column_norms(matrix):
    for i in range(matrix.shape[1]):
        norm = np.linalg.norm(np.array(matrix[:, i], dtype='float'))
        if norm > 2:
            matrix[:, i] /= int(np.floor(norm))
    return matrix


def remove_fake_ecms(R, network):
    metabolite_ids = [network.metabolites[i].id for i in network.external_metabolite_indices()]
    real_ecms = np.array([not fake_ecm(R[:, i], metabolite_ids) for i in range(R.shape[1])])
    return R[:, real_ecms]


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


def intersect_directly(R, internal_metabolites, network, perturbed=False, verbose=True, tol=1e-12):
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
        # i = internal[len(internal)-1]
        to_remove = i - len(deleted[deleted < i])
        if verbose:
            print("\nIteration %d (internal metabolite = %d: %s) of %d" % (it, to_remove, [m.id for m in network.metabolites][to_remove], len(internal_metabolites)))
            print("Possible LP amounts for this step:\n" + ", ".join(np.sort(
                [np.sum(R[j - len(deleted[deleted < j]), :] > 0) * np.sum(R[j - len(deleted[deleted < j]), :] < 0) for j
                 in internal]).astype(str)))
            print("Total: %d" % sum(
                [np.sum(R[j - len(deleted[deleted < j]), :] > 0) * np.sum(R[j - len(deleted[deleted < j]), :] < 0) for j
                 in internal]))
            it += 1
        R, removed = eliminate_metabolite(R, i - len(deleted[deleted < i]), network, calculate_adjacency=True,
                                          perturbed=perturbed)
        rows_removed_redund += removed
        deleted = np.append(deleted, i)
        internal.remove(i)

    # remove artificial rays introduced by splitting metabolites
    # R = remove_fake_ecms(R, network)
    R, ids = unsplit_metabolites(R, network)

    if verbose:
        print("\n\tRows removed by redund overall: %d\n" % rows_removed_redund)
        if rows_removed_redund != 0:
            pass
            # input("Waiting...")

    return R, ids
