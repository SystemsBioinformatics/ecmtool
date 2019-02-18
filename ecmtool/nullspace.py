from scipy import linalg
import numpy as np
from .helpers import nullspace, to_fractions


def amount_nonzero_diagonals(R):
    width = R.shape[1]
    nonzero = 0
    for i in range(width):
        if R[i,i] != 0:
            nonzero += 1
    return nonzero


def qr_nullspace(A):
    Q, R, P = linalg.qr(A, pivoting=True, check_finite=False)
    rank = amount_nonzero_diagonals(R)
    nullspace_basis = Q[:, -rank:]

    return nullspace_basis


def nullspace_symbolic(A):
    return nullspace(A, symbolic=True)


def iterative_nullspace(A, rows_per_iteration = 10, nullspace_method=nullspace_symbolic, verbose=False):
    """
    Based on https://mathematica.stackexchange.com/a/6778
    :param A:
    :param rows_per_iteration:
    :param nullspace_method:
    :return:
    """
    number_partitions = int(np.ceil(A.shape[0] / float(rows_per_iteration)))

    # Pad with zero rows to allow equal split sizes
    needed_rows = rows_per_iteration * number_partitions
    current_rows = A.shape[0]
    if needed_rows > current_rows:
        A = np.append(A, to_fractions(np.zeros(shape=(needed_rows - current_rows, A.shape[1]))), axis=0)

    partitions = np.split(A, number_partitions)

    if verbose:
        print('Calculating initial nullspace basis')

    # Begin with the nullspace of the topmost partition
    partial_nullspace = nullspace_method(partitions[0])

    # Every iteration decreases the partial nullspace
    for round in range(1, len(partitions)):
        if verbose:
            print('Calculating partial nullspace basis %d/%d' % (round, len(partitions) - 1))

        if partial_nullspace.shape[1] == 0:
            return to_fractions(np.ndarray(shape=(A.shape[1], 0)))

        current = partitions[round]
        multipliers = nullspace_method(np.dot(current, partial_nullspace))
        if multipliers.shape[1] == 0:
            return to_fractions(np.ndarray(shape=(A.shape[1], 0)))
        partial_nullspace = np.transpose(np.dot(np.transpose(multipliers), np.transpose(partial_nullspace)))

    # Now the partial nullspace is the full nullspace
    return partial_nullspace


if __name__ == '__main__':
    null = iterative_nullspace(np.asarray([[-1, 1, 0], [0, -1, 1]], dtype=int), rows_per_iteration=1, verbose=True)
    pass