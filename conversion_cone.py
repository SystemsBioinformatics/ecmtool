import numpy as np
from helpers import *
from sympy import *
from time import time


def get_conversion_cone(N, tagged_rows=[], reversible_columns=[], verbose=False):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).
    :param N: stoichiometry matrix
    :param tagged_rows: list of row numbers (0-based) of metabolites that are tagged as in/outputs ("conversions")
    :param reversible_columns: list of booleans stating whether the reaction at this column is reversible
    :return: matrix with conversion cone "c" as row vectors
    """
    G = np.transpose(N)

    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_columns:
            G = np.append(G, [-G[reaction_index, :]], axis=0)

    # We use sympy.Matrix so we can calculate the null space in an exact fashion.
    # Linealities are a biological synonym of the null space, and describe the degrees
    # of freedom that our generator G has.
    if verbose:
        print('Calculating nullspace of G')
    # G_linealities = np.transpose(nullspace(G))
    nullspace_vectors = Matrix(G).nullspace()
    nullspace = nullspace_vectors[0].T
    for i in range(1, len(nullspace_vectors)):
        nullspace = nullspace.row_insert(-1, nullspace_vectors[i].T)

    G_linealities = to_fractions(np.asarray(nullspace.rref()[0], dtype='object'))
    G_linealities = np.append(G_linealities, -G_linealities, axis=0)

    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    H_cone = np.asarray(list(get_extreme_rays(None, G, verbose=verbose)))
    H = np.append(G_linealities, H_cone, axis=0) if G_linealities.shape[0] else H_cone

    if verbose:
        print('Appending constraint B == 0')
    # Append B == 0 constraint
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]
    row_tags = np.zeros(shape=(amount_metabolites * 2, amount_metabolites))

    for reaction_index in range(amount_metabolites):
        if reaction_index not in tagged_rows:
            row_tags[reaction_index, reaction_index] = 1
            row_tags[(reaction_index + amount_metabolites), reaction_index] = -1

    H_mod = np.append(H, row_tags, axis=0)
    # H_mod = H


    if verbose:
        print('Appending constraint biomass == 1')
    H_bio = np.append(np.zeros(shape=(H_mod.shape[0], 1)), H_mod, axis=1)
    H_bio = np.append(H_bio, np.zeros(shape=(2, H_bio.shape[1])), axis=0)
    H_bio[-2, 0] = -1 # -1 * Target biomass concentration change
    H_bio[-2, -1] = 1 # Biomass metabolite
    H_bio[-1, 0] = 1 # Target biomass concentration change
    H_bio[-1, -1] = -1 # -1 Biomass metabolite

    if verbose:
        print('Calculating extreme rays C of inequalities system H')
    rays = np.asarray(list(get_extreme_rays(None, H_bio, verbose=verbose)))

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays, H_cone, H

    # Normalise rays
    for row in range(rays.shape[0]):
        rays[row,:] /= rays[row, 0] if rays[row, 0] != 0 else 1
    rays = rays[:, 1:] # Drop the added column

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
