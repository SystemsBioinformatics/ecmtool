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

    for metabolite_index in range(G.shape[0]):
        if metabolite_index in reversible_columns:
            G = np.append(G, [-G[metabolite_index, :]], axis=0)

    # We use sympy.Matrix so we can calculate the null space in an exact fashion.
    # Linealities are a biological synonym of the null space, and describe the degrees
    # of freedom that our generator G has.
    if verbose:
        print('Calculating nullspace of G')
    # G_linealities = np.transpose(nullspace(G))
    G_linealities = np.asarray(Matrix(G).nullspace(), dtype='float')
    G_linealities = np.append(G_linealities, -G_linealities, axis=0)

    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    H_cone = np.asarray(list(get_extreme_rays(None, G, verbose=verbose)))
    H = np.append(G_linealities, H_cone, axis=0) if G_linealities.shape[0] else H_cone

    if verbose:
        print('Appending constraint B == 0')
    # Append B == 0 constraint
    amount_metabolites = N.shape[0]
    row_tags = np.zeros(shape=(amount_metabolites * 2, amount_metabolites))

    for metabolite_index in range(amount_metabolites):
        if metabolite_index not in tagged_rows:
            row_tags[metabolite_index, metabolite_index] = 1
            row_tags[(metabolite_index + amount_metabolites), metabolite_index] = -1

    H_mod = np.append(H, row_tags, axis=0)
    # H_mod = H

    if verbose:
        print('Calculating extreme rays C of inequalities system H')
    rays = np.asarray(list(get_extreme_rays(None, H_mod, verbose=verbose)))

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')

    return rays


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
