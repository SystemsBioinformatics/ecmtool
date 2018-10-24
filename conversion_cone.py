import numpy as np
from helpers import *
from sympy import *
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
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_columns:
            G = np.append(G, [-G[reaction_index, :]], axis=0)

    if verbose:
        print('Calculating nullspace of G')
    # Add (reduced row echelon form of) the nullspace matrix to our linealities in positive and negative direction
    # Linealities are a biological synonym of the null space, and describe the degrees
    # of freedom that our generator G has.
    G_deflated = deflate_matrix(G, tagged_rows)
    G_linealities_deflated = nullspace(G_deflated, symbolic=symbolic)
    G_linealities_deflated = np.append(G_linealities_deflated, -G_linealities_deflated, axis=0)
    G_linealities = inflate_matrix(G_linealities_deflated, tagged_rows, G.shape[1])

    # Calculate H as the union of our linealities and the extreme rays of matrix G (all as row vectors)
    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    # H_cone = np.asarray(list(get_extreme_rays(None, G, verbose=verbose)))
    # H_cone2 = np.asarray(list(get_extreme_rays(None, np.append(G, G_linealities, axis=0), verbose=verbose)))
    H_cone = np.asarray(list(get_extreme_rays(None, np.append(G, G_linealities, axis=0), fractional=symbolic, verbose=verbose)))
    # H_cone = normalize_rows(H_cone)

    H = np.append(G_linealities, H_cone, axis=0) if G_linealities.shape[0] else H_cone

    # Create constraints that internal metabolites shouldn't change over time
    if verbose:
        print('Appending constraint B == 0')
    row_tags = np.zeros(shape=(amount_metabolites * 2, amount_metabolites))

    for reaction_index in range(amount_metabolites):
        if reaction_index not in tagged_rows:
            row_tags[reaction_index, reaction_index] = 1
            row_tags[(reaction_index + amount_metabolites), reaction_index] = -1

    # Create constraints that input metabolites should have seminegative change over time,
    # and outputs semipositive.
    if verbose:
        print('Appending constraint inputs <= 0, outputs >= 0')
    for index in input_metabolites:
        row = np.zeros(shape=(1, amount_metabolites))
        row[0, index] = -1
        row_tags = np.append(row_tags, row, axis=0)
    for index in output_metabolites:
        row = np.zeros(shape=(1, amount_metabolites))
        row[0, index] = 1
        row_tags = np.append(row_tags, row, axis=0)

    # Append above additional constraints to the final version of H
    H_mod = np.append(H, row_tags, axis=0)

    if verbose:
        print('Calculating nullspace of H')
        # Add (reduced row echelon form of) the nullspace matrix to our linealities in positive and negative direction
    H_linealities = nullspace(H_mod, symbolic=symbolic)
    H_linealities = np.append(H_linealities, -H_linealities, axis=0)

    # Calculate the extreme rays of this cone H, resulting in the elementary
    # conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H')
    rays = np.asarray(list(get_extreme_rays(None, np.append(H_mod, H_linealities, axis=0), fractional=symbolic, verbose=verbose)))

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays, H_cone, H

    return rays, H_cone, H


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
