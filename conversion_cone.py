import numpy as np
from helpers import *
from sympy import Matrix
from time import time


def normalize_rows(M):
    row_max = M.max(axis=1)
    return M / np.transpose(np.asarray(np.asmatrix(row_max, dtype='object'), dtype='object'))

def get_rownames(A):
    rownames = []
    for row_index in range(A.shape[0]):
        rownames.append([index for index, value in enumerate(A[row_index, :]) if value != 0])
    return rownames


def deflate_matrix(A, columns_to_keep):
    B = np.ndarray(shape=(0, len(columns_to_keep)), dtype=A.dtype)

    # Return rows that are nonzero after removing unwanted columns
    for row_index in range(A.shape[0]):
        row = A[row_index, columns_to_keep]
        if np.sum(row) > 0:
            B = np.append(B, [row], axis=0)

    return B

def inflate_matrix(A, kept_columns, original_width):
    B = np.zeros(shape=(A.shape[0], original_width), dtype=A.dtype)

    for index, col in enumerate(kept_columns):
        B[:, col] = A[:, index]

    return B


def de_groot_nullbase(A):
    """
    Calculates De Groot base(TM)(R) of a nullspace matrix.
    :param A: nullspace matrix
    :return:
    """
    A = to_fractions(A)
    k_rows, n_cols = A.shape
    rownames = get_rownames(A)
    steps = ['%d' % i for i in range(k_rows)]


    # for m in range(1, n_cols - k_rows + 1):
    #     cur_iteration_rows = A.shape[0]
    #     for row_index in range(A.shape[0]):
    #
    #         if A[row_index, k_rows + m-1] == 0 or \
    #            rownames[row_index][m-1] > k_rows + m -1:
    #             continue
    #
    #         # Normalise row
    #         # TODO: add support for columns with zero value
    #         A[row_index, :] /= A[row_index, k_rows + m - 1]
    #         cur_row = A[row_index, :]
    #
    #         # All rows except the current row
    #         other_rows = np.setdiff1d(range(cur_iteration_rows), [row_index])
    #         for other_row_index in other_rows:
    #             cur_rownames = rownames[row_index]
    #             other_rownames = rownames[other_row_index]
    #
    #             # j > im && j <= k + m - 1
    #             if len(np.union1d([row for row in cur_rownames if row <= k_rows + m - 2], [row for row in other_rownames if row <= k_rows + m - 2])) <= (m + 1) and \
    #                             other_rownames[m-1] > cur_rownames[m-1] and other_rownames[m-1] <= k_rows + m - 2:
    #                 # TODO: add support for columns with zero value
    #                 other_row = A[other_row_index, :]
    #                 # Normalise other row
    #                 other_row /= other_row[k_rows + m - 1]
    #                 A = np.append(A, np.asarray(np.asmatrix(np.subtract(cur_row, other_row), dtype='object'), dtype='object'), axis=0)
    #                 steps.append('%d => %d - %d' % (k_rows + m - 1, row_index, other_row_index))
    #
    #             # Has to be recalculated after every new row has been added
    #             # TODO: can of course be made tremendously more efficient
    #             rownames = get_rownames(A)
    #
    # return A

    for m in range(1, n_cols - k_rows + 1):
        cur_iteration_rows = A.shape[0]
        for row_index in range(cur_iteration_rows):

            if A[row_index, k_rows + m - 1] == 0:
                continue

            # Normalise row
            # TODO: add support for columns with zero value
            A[row_index, :] /= A[row_index, k_rows + m - 1]
            cur_row = A[row_index, :]

            for other_row_index in range(row_index+1, cur_iteration_rows):
                cur_rownames = [row for row in rownames[row_index] if row <= k_rows + m - 2]
                other_rownames = [row for row in rownames[other_row_index] if row <= k_rows + m - 2]

                if len(np.union1d(cur_rownames, other_rownames) <= (m + 1)):
                    #TODO: add support for columns with zero value
                    other_row = A[other_row_index, :]

                    if other_row[k_rows + m - 1] == 0:
                        continue

                    # Normalise other row
                    other_row /= other_row[k_rows + m - 1]
                    A = np.append(A, np.asarray(np.asmatrix(np.subtract(cur_row, other_row), dtype='object'), dtype='object'), axis=0)
                    steps.append('%d => %d - %d' % (k_rows + m - 1, row_index, other_row_index))

            rownames = get_rownames(A)

    return A


def get_conversion_cone(N, tagged_rows=[], reversible_columns=[], input_metabolites=[], output_metabolites=[],
                        symbolic=True, verbose=False):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).
    :param N: stoichiometry matrix
    :param tagged_rows: list of row numbers (0-based) of metabolites that are tagged as in/outputs ("conversions")
    :param reversible_columns: list of booleans stating whether the reaction at this column is reversible
    :return: matrix with conversion cone "c" as row vectors
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]

    # Compose G of the columns of N
    G = np.transpose(N)

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for metabolite_index in range(G.shape[0]):
        if metabolite_index in reversible_columns:
            G = np.append(G, [-G[metabolite_index, :]], axis=0)

    # Add external metabolites in both directions
    G = np.append(G, -G[:, tagged_rows], axis=1)
    extra_metabolites = [index + amount_metabolites for index in range(len(tagged_rows))]

    # Remove negative entries
    for row in range(G.shape[0]):
        for col in (tagged_rows + extra_metabolites):
            G[row, col] = max(0, G[row, col])

    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    rays = get_extreme_rays_cdd(G)
    H_ineq = np.ndarray(shape=(0, rays.shape[1]))
    linearities = np.ndarray(shape=(0, rays.shape[1]))

    for row in range(rays.shape[0]):
        if np.all(np.dot(G, rays[row, :]) == 0):
            # This is a lineality
            linearities = np.append(linearities, [rays[row, :]], axis=0)
        else:
            # This is a normal extreme ray
            H_ineq = np.append(H_ineq, [rays[row, :]], axis=0)


    H_eq = linearities

    # Create constraints that internal metabolites shouldn't change over time
    if verbose:
        print('Appending constraint B == 0')
    constraints = np.ndarray(shape=(0, rays.shape[1]))
    for internal_metabolite in np.setdiff1d(range(amount_metabolites), tagged_rows):
        equality = to_fractions(np.asarray([[0] * internal_metabolite + [1] + [0] * (amount_metabolites + len(tagged_rows) - (internal_metabolite + 1))]))
        H_eq = np.append(H_eq, equality, axis=0)

    if verbose:
        print('Appending constraint c >= 0')
    semipositivity = np.identity(H_ineq.shape[1])
    H_ineq = np.append(H_ineq, to_fractions(semipositivity), axis=0)

    if verbose:
        print('Calculating nullspace A of H_eq')
    A = np.transpose(nullspace(H_eq))

    # Append above additional constraints to the final version of H
    H_total = np.dot(H_ineq, A)

    # Calculate the extreme rays of this cone H, resulting in the elementary
    # conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H_total')
    rays_full = np.transpose(np.dot(A, np.transpose(np.asarray(list(get_extreme_rays(None, H_total, fractional=symbolic, verbose=verbose))))))
    # rays_full = np.transpose(np.dot(A, np.transpose(get_extreme_rays_cdd(H_total))))

    if rays_full.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays_full, H_ineq

    # Merge the negative exchange metabolite directions with their original again
    rays_full[:, tagged_rows] = np.subtract(rays_full[:, tagged_rows], rays_full[:, amount_metabolites:])
    rays_compact = rays_full[:, :amount_metabolites]

    # TODO: find out how to prevent null vectors caused by S - S_neg = 0 ray
    rays_compact = rays_compact[np.unique(np.nonzero(rays_compact)[0]), :]

    return rays_compact, H_ineq


if __name__ == '__main__':
    start = time()

    de_groot_nullbase(np.asarray([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 3, 2],
        [0, 0, 1, 5, 3],
    ]))

    S = np.asarray([
        [-1, 0, 0],
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, -1],
        [0, 0, 1]])
    c = get_conversion_cone(S, [0, 1, 4], verbose=True)
    print(c)

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
