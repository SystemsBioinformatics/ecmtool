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

    # Add metabolites in both directions
    G = np.append(G, -G, axis=1)

    # Remove negative entries
    for row in range(G.shape[0]):
        for col in range(G.shape[1]):
            G[row, col] = max(0, G[row, col])

    # Calculate H as the union of our linealities and the extreme rays of matrix G (all as row vectors)
    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    H = get_extreme_rays_cdd(G)

    for row in range(H.shape[0]):
        if np.all(np.dot(G, H[row, :]) == 0):
            # This is a lineality
            H = np.append(H, [-H[row, :]], axis=0)


    constraints = np.ndarray(shape=(0, H.shape[1]))

    # Create constraints that internal metabolites shouldn't change over time
    if verbose:
        print('Appending constraint B == 0')
    metabolite_identity = np.identity(H.shape[1] / 2)

    for external_metabolite in tagged_rows:
        metabolite_identity[external_metabolite] = 0

    internal_constraint = np.append(metabolite_identity, -metabolite_identity, axis=1)
    internal_constraint = np.append(internal_constraint, -internal_constraint, axis=0)
    constraints = np.append(constraints, internal_constraint, axis=0)

    if verbose:
        print('Appending constraint c >= 0')
    semipositivity = np.identity(H.shape[1])
    constraints = np.append(constraints, semipositivity, axis=0)

    # Append above additional constraints to the final version of H
    H_constrained = np.append(H, constraints, axis=0)

    # Calculate the extreme rays of this cone H, resulting in the elementary
    # conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H')
    # rays_full = np.asarray(list(get_extreme_rays(None, H_constrained, fractional=symbolic, verbose=verbose)))
    rays_full = get_extreme_rays_cdd(H_constrained)

    if rays_full.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays_full, H

    # Merge the negative exchange metabolite directions with their original again
    rays_compact = np.subtract(rays_full[:, 0:amount_metabolites], rays_full[:, amount_metabolites:])

    return rays_compact, H


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
