from os import system

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
        if np.count_nonzero(row) > 0:
            B = np.append(B, [row], axis=0)

    return B


def inflate_matrix(A, kept_columns, original_width):
    B = np.zeros(shape=(A.shape[0], original_width), dtype=A.dtype)

    for index, col in enumerate(kept_columns):
        B[:, col] = A[:, index]

    return B


def redund(matrix, verbose=False):
    matrix = to_fractions(matrix)

    with open('tmp/matrix.ine', 'w') as file:
        file.write('V-representation\n')
        file.write('begin\n')
        file.write('%d %d rational\n' % (matrix.shape[0], matrix.shape[1] + 1))
        for row in range(matrix.shape[0]):
            file.write(' 0')
            for col in range(matrix.shape[1]):
                file.write(' %s' % str(matrix[row, col]))
            file.write('\n')
        file.write('end\n')

    system('scripts/redund tmp/matrix.ine > tmp/matrix_nored.ine')

    matrix_nored = np.ndarray(shape=(0, matrix.shape[1] + 1), dtype='object')

    with open('tmp/matrix_nored.ine') as file:
        lines = file.readlines()
        for line in [line for line in lines if line not in ['\n', '']]:
            # Skip comment and INE format lines
            if np.any([target in line for target in ['*', 'V-representation', 'begin', 'end', 'rational']]):
                continue
            row = [Fraction(x) for x in line.replace('\n', '').split(' ') if x != '']
            matrix_nored = np.append(matrix_nored, [row], axis=0)

    remove('tmp/matrix.ine')
    remove('tmp/matrix_nored.ine')

    if verbose:
        print('Removed %d redundant rows' % (matrix.shape[0] - matrix_nored.shape[0]))

    return matrix_nored[:, 1:]


def get_zero_set(G, equalities):
    dot_products = np.transpose(np.asarray([np.dot(G, equality) for equality in equalities]))
    zero_set = [set([index for index, value in enumerate(row) if value == 0]) for row in dot_products]
    return zero_set


def drop_nonextreme(G, zero_set, verbose=False):
    keep_rows = []

    for row in range(G.shape[0]):
        remove = False
        for other_row in range(G.shape[0]):
            # The current row (reaction) has a zero set that is
            # a proper subset of another row's. This means the other
            # row represents a more extreme vector in the same plane.
            if zero_set[row] < zero_set[other_row]:
                remove = True
                break

        if not remove:
            keep_rows.append(row)


    if verbose:
        print('Removing %d nonextreme rays' % (G.shape[0] - len(keep_rows)))

    return G[keep_rows, :]


def get_clementine_conversion_cone(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                                   verbose=True):
    """
    Calculates the conversion cone using Superior Clementine Equality Intersection (all rights reserved).
    Follows the general Double Description method by Motzkin, using G as initial basis and intersecting
    hyperplanes of internal metabolites = 0.
    :param N:
    :param external_metabolites:
    :param reversible_reactions:
    :param input_metabolites:
    :param output_metabolites:
    :return:
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]
    internal_metabolites = np.setdiff1d(range(amount_metabolites), external_metabolites)

    identity = np.identity(amount_metabolites)
    equalities = [identity[:, index] for index in internal_metabolites]

    # Compose G of the columns of N
    G = np.transpose(N)

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_reactions:
            G = np.append(G, [-G[reaction_index, :]], axis=0)

    # For each internal metabolite, intersect the intermediary cone with an equality to 0 for that metabolite
    for index, internal_metabolite in enumerate(internal_metabolites):
        if verbose:
            print('Iteration %d/%d' % (index, len(internal_metabolites)))

        # Find conversions that use this metabolite
        active_conversions = np.asarray([conversion_index for conversion_index in range(G.shape[0])
                              if G[conversion_index, internal_metabolite] != 0])

        # Skip internal metabolites that aren't used anywhere
        if len(active_conversions) == 0:
            if verbose:
                print('Skipping internal metabolite #%d, since it is not used by any reaction\n' % internal_metabolite)
            continue

        # Project conversions that use this metabolite onto the hyperplane internal_metabolite = 0
        projections = np.dot(G[active_conversions, :], equalities[index])
        positive = active_conversions[np.argwhere(projections > 0)[:, 0]]
        negative = active_conversions[np.argwhere(projections < 0)[:, 0]]
        candidates = np.ndarray(shape=(0, amount_metabolites))

        if verbose:
            print('Adding %d candidates' % (len(positive) * len(negative)))

        # Make conic combinations of all pairs (positive, negative) such that their internal_metabolite = 0
        for pos in positive:
            for neg in negative:
                candidate = np.add(G[pos, :], G[neg, :] * (G[pos, internal_metabolite] / -G[neg, internal_metabolite]))
                candidates = np.append(candidates, [candidate], axis=0)

        # Keep only rays that satisfy internal_metabolite = 0
        keep = np.setdiff1d(range(G.shape[0]), np.append(positive, negative, axis=0))
        if verbose:
            print('Removing %d rays\n' % (G.shape[0] - len(keep)))
        G = G[keep, :]
        G = np.append(G, candidates, axis=0)
        # G = drop_nonextreme(G, get_zero_set(G, equalities), verbose=verbose)
        G = redund(G, verbose=verbose)

    return G


def split_columns_semipositively(matrix, columns):
    orig_column_count = matrix.shape[1]
    matrix = split_columns(matrix, columns)
    semipositive_columns = columns + [orig_column_count + index for index in range(len(columns))]

    for row in range(matrix.shape[0]):
        for column in semipositive_columns:
            matrix[row, column] = max(matrix[row, column], 0)

    return matrix


def split_columns(matrix, columns):
    matrix = np.append(matrix, -matrix[:, columns], axis=1)
    return matrix


def get_conversion_cone(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                        symbolic=True, verbose=False):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).
    :param N: stoichiometry matrix
    :param external_metabolites: list of row numbers (0-based) of metabolites that are tagged as in/outputs ("conversions")
    :param reversible_reactions: list of booleans stating whether the reaction at this column is reversible
    :return: matrix with conversion cone "c" as row vectors
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]

    # External metabolites that have no direction specified
    in_out_metabolites = np.setdiff1d(external_metabolites, np.append(input_metabolites, output_metabolites, axis=0))
    added_virtual_metabolites = np.asarray(np.add(range(len(in_out_metabolites)), amount_metabolites), dtype='int')
    extended_external_metabolites = np.append(external_metabolites, added_virtual_metabolites, axis=0)
    in_out_indices = [external_metabolites.index(index) for index in in_out_metabolites]

    # Compose G of the columns of N
    G = np.transpose(N)

    # TODO: remove debug block
    # G = np.asarray(G * 10**6, dtype='int')

    G_exp = G[:,:]
    G_rev = np.ndarray(shape=(0, G.shape[1]), dtype='object')
    G_irrev = np.ndarray(shape=(0, G.shape[1]), dtype='object')

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_reactions:
            G_exp = np.append(G_exp, [-G[reaction_index, :]], axis=0)
            G_rev = np.append(G_rev, [-G[reaction_index, :]], axis=0)
        else:
            G_irrev = np.append(G_irrev, [G[reaction_index, :]], axis=0)


    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
         print('Calculating null space of inequalities system G')
    linearities = np.transpose(nullspace(G, symbolic=symbolic))
    # linearities = np.transpose(nullspace_terzer(G, verbose=verbose))
    # linearities = nullspace_polco(G, verbose=verbose)
    # linearities = np.loadtxt("/tmp/lin_ecoli2.csv", delimiter=',', dtype='int')
    if linearities.shape[0] == 0:
        linearities = np.ndarray(shape=(0, G.shape[1]))

    # if symbolic and linearities.shape[0] > 0:
    #     assert np.sum(np.sum(np.dot(G, np.transpose(linearities)), axis=0)) == 0
    # elif linearities.shape[0] > 0:
    #     sum = np.sum(np.sum(np.dot(G, np.transpose(linearities)), axis=0))
    #     assert -10 ** -6 <= sum <= 10 ** -6
    linearities_deflated = deflate_matrix(linearities, external_metabolites)

    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
         print('Calculating extreme rays H of inequalities system G')

    # Calculate generating set of the dual of our initial conversion cone C0, C0*
    rays = np.asarray(list(get_extreme_rays(np.append(linearities, G_rev, axis=0), G_irrev, verbose=verbose, symbolic=symbolic)))
    # rays = np.asarray(list(get_extreme_rays(None, G_exp, verbose=verbose, symbolic=symbolic)))
    rays = rays if rays.shape[0] else np.ndarray(shape=(0, G.shape[1]))
    rays_deflated = deflate_matrix(rays, external_metabolites)

    # Add bidirectional (in- and output) metabolites in reverse direction
    rays_split = split_columns(rays_deflated, in_out_indices)
    linearities_split = split_columns(linearities_deflated, in_out_indices)

    H_ineq = rays_split
    H_eq = linearities_split

    # Add input/output constraints to H_ineq
    if not H_ineq.shape[0]:
        H_ineq = np.zeros(shape=(1, H_ineq.shape[1]))

    identity = np.identity(H_ineq.shape[1])

    # Bidirectional (in- and output) metabolites
    for list_index, inout_metabolite_index in enumerate(in_out_indices):
        index = inout_metabolite_index
        H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)
        index = len(external_metabolites) + list_index
        H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

    # Inputs
    for input_metabolite in input_metabolites:
        index = external_metabolites.index(input_metabolite)
        H_ineq = np.append(H_ineq, [-identity[index, :]], axis=0)

    # Outputs
    for output_metabolite in output_metabolites:
        index = external_metabolites.index(output_metabolite)
        H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

    # Calculate the extreme rays of the cone C represented by inequalities H_total, resulting in
    # the elementary conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H_eq, H_ineq')

    # rays = np.asarray(list(get_extreme_rays_efmtool(H_total)))
    # rays = np.asarray(list(get_extreme_rays(None, H_total, verbose=verbose)))
    rays = np.asarray(list(get_extreme_rays(H_eq if len(H_eq) else None, H_ineq, verbose=verbose, symbolic=symbolic)))
    # rays = get_extreme_rays_cdd(H_total)

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays

    if verbose:
        print('Inflating rays')
    rays_inflated = inflate_matrix(rays, extended_external_metabolites, amount_metabolites + len(in_out_metabolites))

    # Merge bidirectional metabolites again, and drop duplicate rows
    # np.unique() requires non-object matrices, so here we cast our results into float64.
    rays_inflated[:, in_out_metabolites] = np.subtract(rays_inflated[:, in_out_metabolites], rays_inflated[:, G.shape[1]:])
    rays_merged = np.asarray(rays_inflated[:, :G.shape[1]], dtype='float64')
    return np.unique(rays_merged, axis=0)


if __name__ == '__main__':
    start = time()

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
