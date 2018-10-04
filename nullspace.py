import numpy as np
from sympy import *


def nullspace_extreme_rays(N, nrev):
    """
    Uses Nullspace algorithm from (Urbanczik, 2004) to compute
    extreme rays (i.e. minimal generating set) of a stoichiometry matrix.
    The matrix should have full row rank, and if there are reversible reactions,
    they should be in the first nrev columns.
    :param N: input matrix as np.ndarray(shape=(m, n))
    :param nrev: the amount of columns that are reversible
    :return: matrix with extreme rays as column vectors
    """
    m, n = N.shape
    Ktot = np.transpose(np.asarray(Matrix(N).nullspace()))
    row_permutations = order_null_space(Ktot, nrev)
    betas = np.identity(n-m)

    # zero-based indexing, so we omit the +1 from (Urbanczik, 2004)
    for j in range((n - m), n):
        newrow = np.dot(Ktot[row_permutations[j],:], betas)
        pass


    pass


def order_null_space(Ktot, nrev):
    # TODO: implement ordering as in Urbanczik
    return range(Ktot.shape[0] + 1)


if __name__ == '__main__':
    N = np.asarray([
    [-1, 0],
    [1, -1],
    [0, 1]])

    rays = nullspace_extreme_rays(N.transpose(), 0)