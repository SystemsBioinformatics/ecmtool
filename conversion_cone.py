import numpy as np
from helpers import *
from sympy import *
from time import time


def get_conversion_cone(N, tagged_rows=[], reversible_columns=[], input_metabolites=[], output_metabolites=[], verbose=False):
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

    # We use sympy.Matrix so we can calculate the null space in an exact fashion.
    # Linealities are a biological synonym of the null space, and describe the degrees
    # of freedom that our generator G has.
    if verbose:
        print('Calculating nullspace of G')
    nullspace_vectors = Matrix(G).nullspace()

    # Add nullspace vectors to a nullspace matrix as row vectors
    nullspace_matrix = nullspace_vectors[0].T if len(nullspace_vectors) else None
    for i in range(1, len(nullspace_vectors)):
        nullspace_matrix = nullspace_matrix.row_insert(-1, nullspace_vectors[i].T)

    # Add (reduced row echelon form of) the nullspace matrix to our linealities in positive and negative direction
    G_linealities = to_fractions(np.asarray(nullspace_matrix.rref()[0], dtype='object')) if nullspace_matrix else np.ndarray(shape=(0, amount_metabolites))
    G_linealities = np.append(G_linealities, -G_linealities, axis=0)

    # Calculate H as the union of our linealities and the extreme rays of matrix G (all as row vectors)
    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    H_cone = np.asarray(list(get_extreme_rays(None, G, verbose=verbose)))
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

    # Calculate the extreme rays of this cone H, resulting in the elementary
    # conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H')
    rays = np.asarray(list(get_extreme_rays(None, H_mod, verbose=verbose)))

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays, H_cone, H

    return rays, H_cone, H


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
