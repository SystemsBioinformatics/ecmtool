import numpy as np
from time import time
from scipy.optimize import linprog

from ecmtool.helpers import redund


def fake_ecm(reaction, metabolite_ids, tol=1e-12):
    s = ""
    for i,c in enumerate(np.asarray(reaction, dtype='int')):
        if abs(reaction[i]) > tol:
            if s == "":
                s = metabolite_ids[i].replace("_in", "").replace("_out", "")
            elif s != metabolite_ids[i].replace("_in", "").replace("_out", ""):
                return False
    return True


def print_ecms_direct(R, external_metabolites, metabolites):
    metabolite_ids = [metabolites[i].id for i in external_metabolites]

    #obj_id = 0
    #for m in external_metabolites:
    #    if metabolites[m].id == "objective":
    #        break
    #    obj_id += 1


    print("\n--ECMs found by intersecting directly--\n")
    count = 0
    for i in range(R.shape[1]):
        if not fake_ecm(R[:,i], metabolite_ids):
            print("ECM #%d:" % count)
            count += 1
    #    div = 1
    #    if R[obj_id][i] != 0:
    #        div = R[obj_id][i]
            for j in range(R.shape[0]):
                if R[j][i] != 0:
                    print("%s: %f" % (metabolite_ids[j].replace("_in","").replace("_out",""), float(R[j][i])))
            print("")


def get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape

    #if (len(basis) > 0 and np.linalg.matrix_rank(A[:,basis]) < len(basis)):
    #    raise Exception("Basis has dependent columns")

    rank = np.linalg.matrix_rank(A[:,basis])
    new_basis = basis.copy()
    for i in range(n):
        if i in new_basis:
            continue
        prev_rank = rank
        prev_basis = new_basis
        new_basis = np.append(new_basis, i)
        rank = np.linalg.matrix_rank(A[:, new_basis])

        if rank == prev_rank: # column added did not increase rank
            new_basis = prev_basis
        if rank == m:
            break

    return new_basis


def kkt_check(c, A, x, basis, tol=1e-12, verbose=True):
    """
    Determine whether KKT conditions hold for x0.
    Take size 0 steps if available.
    """

    ab = np.arange(A.shape[0])
    a = np.arange(A.shape[1])

    th_star = 0
    it = 0
    basis = np.sort(basis)
    basis_hashes = {hash(basis.tostring())} # store hashes of bases used before to prevent cycling
    while th_star < tol:
        it += 1
        if it > 50:
            print("Cycling?")
            return True
        bl = np.zeros(len(a), dtype=bool)
        bl[basis] = 1
        xb = x[basis]
        n = [i for i in range(len(c)) if i not in basis]
        B = A[:, basis]
        N = A[:, n]
        if np.linalg.matrix_rank(B) < min(B.shape):
            print("\nB became singular!\n")
            return True
        l = np.linalg.solve(np.transpose(B), c[basis])
        sn = c[n] - np.dot(np.transpose(N), l)

        if np.all(sn >= -tol):
            if verbose:
                print("Did %d steps in kkt_check2" % (it-1))
            return True

        j = a[~bl][np.argmin(sn)]
        u = np.linalg.solve(B, A[:, j])

        i = u > tol  # if none of the u are positive, unbounded
        if not np.any(i):
            print("Warning: unbounded problem in KKT_check")
            if verbose:
                print("Did %d steps in kkt_check2" % it - 1)
            return True

        th = xb[i] / u[i]
        l = np.argmin(th)  # implicitly selects smallest subscript
        th_star = th[l]  # step size
        if th_star > tol:
            if verbose:
                print("Did %d steps in kkt_check2" % (it - 1))
            return False

        original = basis[ab[i][l]]
        basis[ab[i][l]] = j
        if np.linalg.matrix_rank(A[:, basis]) < min(A[:, basis].shape):
            basis[ab[i][l]] = original
            entering_options = a[~bl][sn < -tol]
            leaving_options = ab[i][th > -tol]

            basis = get_nonsingular_pair(A, basis, entering_options, leaving_options, basis_hashes)
        basis = np.sort(basis)
        basis_hashes.add(hash(basis.tostring()))


def get_nonsingular_pair(A, basis, entering, leaving, basis_hashes):
   for enter in entering:
       for leave in leaving:
           original = basis[leave]
           basis[leave] = enter
           if np.linalg.matrix_rank(A[:, basis]) >= min(A[:, basis].shape):
               # if hash(np.sort(basis).tostring()) in basis_hashes:
               #     basis[leave] = original
               #     continue
               return basis
           basis[leave] = original
   print("Did not find non-singular entering+leaving index...")
   basis[leaving[0]] = entering[0]
   return basis


def independent_rows(A):
    m, n = A.shape
    basis = np.asarray([0])
    A_float = np.asarray(A, dtype='float')
    rank = np.linalg.matrix_rank(A_float)
    original_rank = rank

    if rank == m:
        return A

    rank = np.linalg.matrix_rank(A_float[basis])
    for i in range(1, m):
        prev_rank = rank
        prev_basis = basis
        basis = np.append(basis, i)
        rank = np.linalg.matrix_rank(A_float[basis])

        if rank == prev_rank: # row added did not increase rank
            basis = prev_basis
        if rank == original_rank:
            break

    return A[basis]


def eliminate_metabolite(R, met, network, calculate_adjacency=True, tol=1e-12, verbose=True):
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
        adj = geometric_ray_adjacency(R, plus=plus, minus=minus, verbose=verbose, remove_cycles=True)

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
            print("Of %d candidates, %d were adjacent (%f percent)" % (
            len(plus) * len(minus), nr_adjacent, 100 * nr_adjacent / (len(plus) * len(minus))))
        else:
            print("Of %d candidates, %d were adjacent (0 percent)" % (len(plus) * len(minus), nr_adjacent))

    next_matrix = np.asarray(next_matrix)

    # redund in case we have too many rows
    if verbose:
       print("\tDimensions before redund: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))
    start = time()
    next_matrix = redund(next_matrix)
    end = time()
    if verbose:
       print("\tDimensions after redund: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))
       print("\tRedund took %f seconds" % (end - start))

    next_matrix = np.transpose(next_matrix)

    # delete all-zero row
    next_matrix = np.delete(next_matrix, met, 0)
    network.drop_metabolites([met])
    print("\tDimensions after deleting row: %d %d" % (next_matrix.shape[0], next_matrix.shape[1]))

    return next_matrix


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
        number_rays = R.shape[1]
        i = 0 + 2 * k
        j = 1 + 2 * k
        if j > R.shape[1] - 1:
            return R, deleted
        ray1 = R[:, i]
        ray2 = R[:, j]
        target = 0.5 * ray1 + 0.5 * ray2

        c = -np.ones(number_rays)
        c[i] = 0
        c[j] = 0
        A_ub = -np.identity(number_rays)
        b_ub = np.zeros(number_rays)
        A_eq = R
        b_eq = target
        x0 = np.zeros(number_rays)
        x0[i] = 0.5
        x0[j] = 0.5

        if sum(abs(target)) < tol:
            augment_reaction = i;
            met = get_remove_metabolite(R, network, augment_reaction)
            if verbose:
                print("Found an unbounded LP, augmenting reaction %d through metabolite %d" % (augment_reaction, met))
            R = eliminate_metabolite(R, met, network, calculate_adjacency=False)

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12}, x0=x0)
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

            if abs(res2.fun) < tol: # res is 'unbounded' but res2 has optimum 0
                break


            augment_reaction = [i for i, val in enumerate(res2.x) if val > 90][0]
            met = get_remove_metabolite(R, network, augment_reaction)
            deleted.append(met)
            if verbose:
                print("Found an unbounded LP, augmenting reaction %d through metabolite %d (%s)" % (augment_reaction, met, network.metabolites[met].id))

            R = eliminate_metabolite(R, met, network, calculate_adjacency=False)
            number_rays = R.shape[1]
            i = 0 + 2 * k
            j = 1 + 2 * k
            ray1 = R[:, i]
            ray2 = R[:, j]
            target = 0.5 * ray1 + 0.5 * ray2

            c = -np.ones(number_rays)
            c[i] = 0
            c[j] = 0
            A_ub = -np.identity(number_rays)
            b_ub = np.zeros(number_rays)
            A_eq = R
            b_eq = target
            x0 = np.zeros(number_rays)
            x0[i] = 0.5
            x0[j] = 0.5

            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12}, x0=x0)
            if res.status == 4:
                print("Numerical difficulties with revised simplex, trying interior point method instead")
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

    return R, deleted


def geometric_ray_adjacency(R, plus=[-1], minus=[-1], quasi_zero_tolerance=1e-8, verbose=True, remove_cycles=True):
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
    # TODO: find better (working) way to remove lin. dep. rows from R_indep
    R_indep = independent_rows(R)

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
    for ind1, i in enumerate(plus):
        for ind2, j in enumerate(minus):
            it = ind2 + ind1 * len(minus)
            if verbose:
                print("Doing LP %d of %d (%.2f percent done)" % (it, total, it * 100 / total))
            ray1 = R_indep[:, i]
            ray2 = R_indep[:, j]
            target = 0.5 * ray1 + 0.5 * ray2

            # set up LP
            c = -np.ones(number_rays)
            c[i] = 0
            c[j] = 0
            A_ub = -np.identity(number_rays)
            b_ub = np.zeros(number_rays)
            A_eq = R_indep
            b_eq = target
            x0 = np.zeros(number_rays)
            x0[i] = 0.5
            x0[j] = 0.5

            if not disable_lp:
                LPs_done += 1
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='revised simplex', options={'tol': 1e-12}, x0=x0)

            if not disable_lp and res.status == 4:
                print("Numerical difficulties with revised simplex, trying interior point method instead")
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point', options={'tol': 1e-12})

            disable_lp = True

            ext_basis = get_more_basis_columns(np.asarray(A_eq, dtype='float'), [i, j])
            KKT = kkt_check(c, np.asarray(A_eq, dtype='float'), x0, ext_basis)
            if not disable_lp and verbose and KKT != (abs(res.fun) < quasi_zero_tolerance):
                print("\n\n!!! error !!!\n\n")
                # print("res.status = %d" % (res.status))
                # print("Adjacent: %s" % (abs(res.fun) < quasi_zero_tolerance))
                # print("KKT (should equal adjacent): %s" % KKT)

            if KKT:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    end = time()
    print("Did %d LPs in %f seconds" % (LPs_done, end - start))
    return adjacency


def intersect_directly(R, internal_metabolites, network, verbose=True, tol=1e-12):
    # rows are rays
    deleted = 0
    iter = 1
    internal = list(internal_metabolites)
    internal.sort()

    for i in np.flip(internal, 0):
        if verbose:
            print("\nIteration %d (internal metabolite = %d) of %d" % (iter, i, len(internal_metabolites)))
            iter += 1
        R = eliminate_metabolite(R, i, network, calculate_adjacency=True)

    return R
#1
