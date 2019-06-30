from ecmtool.helpers import *
from sympy import *
from time import time


def nullspace(N, atol=1e-13, rtol=0):
    """
    Calculates the null space of given matrix N.
    Source: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    :param N: ndarray
            A should be at most 2-D.  A 1-D array with length k will be treated
            as a 2-D with shape (1, k)
    :param atol: float
            The absolute tolerance for a zero singular value.  Singular values
            smaller than `atol` are considered to be zero.
    :param rtol: float
            The relative tolerance.  Singular values less than rtol*smax are
            considered to be zero, where smax is the largest singular value.
    :return: If `A` is an array with shape (m, k), then `ns` will be an array
            with shape (k, n), where n is the estimated dimension of the
            nullspace of `A`.  The columns of `ns` are a basis for the
            nullspace; each element in numpy.dot(A, ns) will be approximately
            zero.
    """
    N = np.atleast_2d(N)
    u, s, vh = svd(N)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def get_conversion_cone(N, tagged_rows=[], reversible_rows=[], verbose=False):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).
    :param N: stoichiometry matrix
    :param tagged_rows: list of row numbers (0-based) of metabolites that are tagged as in/outputs ("conversions")
    :param reversible_rows: list of booleans stating whether the reaction at this column is reversible
    :return: matrix with conversion cone "c" as row vectors
    """
    G = np.transpose(N)

    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_rows:
            G = np.append(G, [-G[reaction_index, :]], axis=0)

    # We use sympy.Matrix so we can calculate the null space in an exact fashion.
    # Linealities are a biological synonym of the null space, and describe the degrees
    # of freedom that our generator G has.
    if verbose:
        print('Calculating nullspace of G')
    G_linealities = np.transpose(nullspace(G))
    G_linealities = np.append(G_linealities, -G_linealities, axis=0)

    if verbose:
        print('Calculating extreme rays H of inequalities system G')
    H = G_linealities
    rays = get_extreme_rays(None, G, verbose=verbose)
    if verbose:
        print('Adding found rays to matrix H')
    for ray in rays:
        H = np.append(H, ray, axis=0)

    if verbose:
        print('Appending constraint B == 0')
    # Append B == 0 constraint
    amount_reactions = G.shape[1]
    row_tags = np.zeros(shape=(amount_reactions * 2, H.shape[1]))

    for reaction_index in range(G.shape[1]):
        if reaction_index not in tagged_rows:
            row_tags[[reaction_index], reaction_index] = 1
            row_tags[[reaction_index + amount_reactions], reaction_index] = -1

    H_mod = np.append(H, row_tags, axis=0)
    # H_mod = H

    if verbose:
        print('Calculating extreme rays C of inequalities system H')
    rays = get_extreme_rays(None, H_mod, verbose=verbose)

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')

    return rays


if __name__ == '__main__':
    start = time()

    S = np.asarray([
        [-1, 0, 0],
        [0, -1, 0],
        [1, 2, -1],
        [0, 0, 1]])
    c = get_conversion_cone(S, [0, 1, 3], [1], verbose=True)

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
