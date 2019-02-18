import os
import sys

import psutil
from fractions import Fraction, gcd
from subprocess import check_call, STDOUT, PIPE
from os import remove, devnull as os_devnull, system

import cdd
import numpy as np
from random import randint

from numpy.linalg import svd
from sympy import Matrix


def relative_path(file_path):
    return os.path.join(os.path.dirname(__file__), file_path)


def open_relative(file_path, mode='r'):
    return open(relative_path(file_path), mode)


def remove_relative(file_path):
    return remove(relative_path(file_path))


def get_total_memory_gb():
    """
    Returns total system memory in GiB (gibibytes)
    :return:
    """
    return psutil.virtual_memory().total / 1024**3


def get_min_max_java_memory():
    """
    Returns plausible starting and maximum virtual memory sizes in gibibytes
    for a java VM, as used to run e.g. Polco. Min is either 10% of system RAM
    or 1 gigabyte, whichever is larger. Max is 80% of system RAM.
    :return:
    """
    total = get_total_memory_gb()
    min = int(np.ceil(float(total) * 0.1))
    max = int(np.round(float(total) * 0.8))
    return min, max


def nullspace_rank_internal(src, dst, verbose=False):
    """
    Translated directly from Polco's NullspaceRank:nullspaceRankInternal().
    :param src: ndarray of Fraction() objects
    :param dst:
    :return:
    """
    rows, cols = src.shape
    row_mapping = [row for row in range(rows)]

    if verbose:
        print('Starting rank calculation')

    for col in range(cols):
        if verbose:
            print('Processing columns (rank) - %.2f%%' % (col / float(cols) * 100))

        row_pivot = col
        if row_pivot >= rows:
            return rows, dst

        pivot_dividend = src[row_pivot, col].numerator
        pivot_divisor = None

        # If pivotDividend == 0, try to find another non-dependent row
        for row in range(row_pivot + 1, rows):
            if pivot_dividend != 0:
                break

            pivot_dividend = src[row, col].numerator
            if pivot_dividend != 0:
                # Swap rows
                src[[row_pivot, row]] = src[[row, row_pivot]]
                dst[[row_pivot, row]] = dst[[row, row_pivot]]

                tmp = row_mapping[row_pivot]
                row_mapping[row_pivot] = row_mapping[row]
                row_mapping[row] = tmp

        if pivot_dividend == 0:
            # Done, col is rank
            # TODO: this is likely wrong. When a column is filled with only zeroes,
            # we need to move on to the next column.
            return col, dst

        pivot_divisor = src[row_pivot, col].denominator

        # Make pivot a 1
        src[row_pivot, :] *= Fraction(pivot_dividend, pivot_divisor)
        dst[row_pivot, :] *= Fraction(pivot_dividend, pivot_divisor)

        for other_row in range(rows):
            if other_row != col:
                # Make it a 0
                other_row_pivot_dividend = src[other_row, col].numerator
                if other_row_pivot_dividend != 0:
                    other_row_pivot_divisor = src[other_row, col].denominator
                    src[other_row, :] -= src[row_pivot, :] * Fraction(other_row_pivot_dividend, other_row_pivot_divisor)
                    dst[other_row, :] -= dst[row_pivot, :] * Fraction(other_row_pivot_dividend, other_row_pivot_divisor)

    return cols, dst


def nullspace_terzer(src, verbose=False):
    src_T = np.transpose(src[:, :])
    dst = to_fractions(np.identity(src_T.shape[0]))
    rank, dst = nullspace_rank_internal(src_T, dst, verbose=verbose)
    len = src_T.shape[0]

    nullspace = to_fractions(np.zeros(shape=(len - rank, dst.shape[1])))

    if verbose:
        print('Starting nullspace calculation')

    for row in range(rank, len):
        if verbose:
            print('Processing rows (rank) - %.2f%%' % ((row - rank) / float(len - rank) * 100))

        sign = 0  # Originally "sgn" in Polco
        scp = 1

        # Find one common multiplicand that makes all cells' fractions integer
        for col in range(len):
            dividend = dst[row, col].numerator
            if dividend != 0:
                divisor = dst[row, col].denominator
                scp /= gcd(scp, divisor)
                scp *= divisor
                sign += np.sign(dividend)

        # We want as many cells in this row to be positive as possible
        if np.sign(scp) != np.sign(sign):
            scp *= -1

        # Scale all cells to integer values
        for col in range(len):
            dividend = dst[row, col].numerator
            value = None

            if dividend == 0:
                value = Fraction(0, 1)
            else:
                divisor = dst[row, col].denominator
                value = dividend * Fraction(scp, divisor)

            nullspace[row - rank, col] = value

    return np.transpose(nullspace)


def nullspace_matlab(N):
    import matlab.engine
    engine = matlab.engine.start_matlab()
    result = engine.null(matlab.double([list(row) for row in N]), 'r')
    x = to_fractions(np.asarray(result))
    return x


def nullspace_polco(A, verbose=False):

    B_neg = np.append(-np.identity(A.shape[1]), np.ones(shape=(A.shape[1], 1)), axis=1)
    B_pos = np.append(np.identity(A.shape[1]), np.ones(shape=(A.shape[1], 1)), axis=1)
    B = np.append(B_neg, B_pos, axis=0)
    A = np.append(A, np.zeros(shape=(A.shape[0], 1)), axis=1)

    result = get_extreme_rays(A, B, verbose=verbose)

    for row in range(result.shape[0]):
        result[row, :] /= 1 if result[row, -1] == 0 else result[row, -1]

    null_vectors = result[:, :-1]
    deduped_vectors = []

    # for row in range(null_vectors.shape[0]):
    #     if not np.any([np.sum(null_vectors[row, :] - entry) == 0 for entry in deduped_vectors]):
    #         deduped_vectors.append(null_vectors[row, :])
    #     else:
    #         print('x')
    #         pass
    #
    # return np.transpose(redund(np.asarray(deduped_vectors, dtype='object')))
    return np.transpose(redund(np.asarray(null_vectors, dtype='object')))


def nullspace(N, symbolic=True, atol=1e-13, rtol=0):
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
    if not symbolic:
        N = np.asarray(N, dtype='int64')
        u, s, vh = svd(N)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj()
        return np.transpose(ns)
    else:
        nullspace_vectors = Matrix(N).nullspace()

        # Add nullspace vectors to a nullspace matrix as row vectors
        # Must be a sympy Matrix so we can do rref()
        nullspace_matrix = nullspace_vectors[0].T if len(nullspace_vectors) else None
        for i in range(1, len(nullspace_vectors)):
            nullspace_matrix = nullspace_matrix.row_insert(-1, nullspace_vectors[i].T)

        return to_fractions(
            np.transpose(np.asarray(nullspace_matrix.rref()[0], dtype='object'))) if nullspace_matrix \
            else np.ndarray(shape=(N.shape[0], 0))


# def get_extreme_rays_efmtool(inequality_matrix, matlab_root='/Applications/MATLAB_R2018b.app/'):
#     H = inequality_matrix
#     r, m = H.shape
#     N = np.append(-np.identity(r), H, axis=1)
#
#     engine = matlab.engine.start_matlab()
#     engine.cd('efmtool')
#     engine.workspace['N'] = matlab.double([list(row) for row in N])
#     engine.workspace['rev'] = ([False] * r) + ([True] * m)
#     result = engine.CalculateFluxModes(matlab.double([list(row) for row in N]), matlab.logical(([False] * r) + ([True] * m)))
#     v = result['efms']
#     x = np.transpose(np.asarray(v)[r:, :])
#     return x

def get_efms(N, reversibility):
    import matlab.engine
    engine = matlab.engine.start_matlab()
    engine.cd('efmtool')
    result = engine.CalculateFluxModes(matlab.double([list(row) for row in N]), matlab.logical(reversibility))
    v = result['efms']
    x = np.transpose(np.asarray(v))
    return x


def get_extreme_rays_cdd(inequality_matrix):
    mat = cdd.Matrix(np.append(np.zeros(shape=(inequality_matrix.shape[0], 1)), inequality_matrix, axis=1), number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    gen = poly.get_generators()
    return np.asarray(gen)[:, 1:]


def get_extreme_rays(equality_matrix=None, inequality_matrix=None, symbolic=True, verbose=False):
    rand = randint(1, 10 ** 6)

    if inequality_matrix is not None and inequality_matrix.shape[0] == 0:
        inequality_matrix = None

    if equality_matrix is not None and equality_matrix.shape[0] == 0:
        equality_matrix = None

    if inequality_matrix is None:
        if equality_matrix is not None:
            # inequality_matrix = np.identity(equality_matrix.shape[1])
            inequality_matrix = np.zeros(shape=(1, equality_matrix.shape[1]))
        else:
            raise Exception('No equality or inequality argument given')

    # if inequality_matrix.shape[1] < 50:
    #     if verbose:
    #         print('Using CDD instead of Polco for enumeration of small system')
    #     ineq = np.append(np.append(equality_matrix, -equality_matrix, axis=0), inequality_matrix, axis=0)
    #     for ray in get_extreme_rays_cdd(ineq):
    #         yield ray
    #     return

    # Write equalities system to disk as space separated file
    if verbose:
        print('Writing equalities to file')
    if equality_matrix is not None:
        with open_relative('tmp' + os.sep + 'eq_%d.txt' % rand, 'w') as file:
            for row in range(equality_matrix.shape[0]):
                file.write(' '.join([str(val) for val in equality_matrix[row, :]]) + '\r\n')

    # Write inequalities system to disk as space separated file
    if verbose:
        print('Writing inequalities to file')
    with open_relative('tmp' + os.sep + 'iq_%d.txt' % rand, 'w') as file:
        for row in range(inequality_matrix.shape[0]):
            file.write(' '.join([str(val) for val in inequality_matrix[row, :]]) + '\r\n')

    # Run external extreme ray enumeration tool
    min_mem, max_mem = get_min_max_java_memory()
    if verbose:
        print('Running polco (%d-%d GiB java VM memory)' % (min_mem, max_mem))

    equality_path = relative_path('tmp' + os.sep + 'eq_%d.txt' % rand)
    inequality_path = relative_path('tmp' + os.sep + 'iq_%d.txt' % rand)
    generators_path = relative_path('tmp' + os.sep + 'generators_%d.txt' % rand)
    with open(os_devnull, 'w') as devnull:
        polco_path = relative_path('polco' + os.sep + 'polco.jar')
        check_call(('java -Xms%dg -Xmx%dg ' % (min_mem, max_mem) +
                    '-jar %s -kind text -sortinput AbsLexMin ' % polco_path +
                    '-arithmetic %s ' % (' '.join(['fractional' if symbolic else 'double'] * 3)) +
                    '-zero %s ' % (' '.join(['NaN' if symbolic else '1e-10'] * 3)) +
                    ('' if equality_matrix is None else '-eq %s ' % equality_path) +
                    ('' if inequality_matrix is None else '-iq %s ' % inequality_path) +
                    '-out text %s' % generators_path).split(' '),
            stdout=(devnull if not verbose else None), stderr=(devnull if not verbose else None))

    # Read resulting extreme rays
    if verbose:
        print('Parsing computed rays')
    with open(generators_path, 'r') as file:
        lines = file.readlines()
        rays = np.ndarray(shape=(0, inequality_matrix.shape[1]))

        if len(lines) > 0:
            number_lines = len(lines)
            number_entries = len(lines[0].replace('\n', '').split('\t'))
            rays = np.repeat(np.repeat(to_fractions(np.zeros(shape=(1,1))), number_entries, axis=1), number_lines, axis=0)

            for row, line in enumerate(lines):
                # print('line %d/%d' % (row+1, number_lines))
                for column, value in enumerate(line.replace('\n', '').split('\t')):
                    if value != '0':
                        rays[row, column] = Fraction(str(value))

    if verbose:
        print('Done parsing rays')

    # Clean up the files created above
    if equality_matrix is not None:
        remove(equality_path)

    remove(inequality_path)
    remove(generators_path)

    return rays


def binary_exists(binary_file):
    return any(
        os.access(os.path.join(path, binary_file), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )


def get_redund_binary():
    if sys.platform.startswith('linux'):
        if not binary_exists('redund'):
            raise EnvironmentError('Executable "redund" was not found in your path. Please install package lrslib (e.g. apt install lrslib)')
        return 'redund'
    elif sys.platform.startswith('win32'):
        return 'redund\\redund_win.exe'
    elif sys.platform.startswith('darwin'):
        return 'redund/redund_mac'
    else:
        raise OSError('Unsupported operating system platform: %s' % sys.platform)


def redund(matrix, verbose=False):
    matrix = to_fractions(matrix)
    binary = relative_path(get_redund_binary())
    matrix_path = relative_path('tmp' + os.sep + 'matrix.ine')
    matrix_nonredundant_path = relative_path('tmp' + os.sep + 'matrix_nored.ine')

    if matrix.shape[0] <= 1:
        return matrix

    with open(matrix_path, 'w') as file:
        file.write('V-representation\n')
        file.write('begin\n')
        file.write('%d %d rational\n' % (matrix.shape[0], matrix.shape[1] + 1))
        for row in range(matrix.shape[0]):
            file.write(' 0')
            for col in range(matrix.shape[1]):
                file.write(' %s' % str(matrix[row, col]))
            file.write('\n')
        file.write('end\n')

    system('%s %s > %s' % (binary, matrix_path, matrix_nonredundant_path))

    matrix_nored = np.ndarray(shape=(0, matrix.shape[1] + 1), dtype='object')

    with open(matrix_nonredundant_path) as file:
        lines = file.readlines()
        for line in [line for line in lines if line not in ['\n', '']]:
            # Skip comment and INE format lines
            if np.any([target in line for target in ['*', 'V-representation', 'begin', 'end', 'rational']]):
                continue
            row = [Fraction(x) for x in line.replace('\n', '').split(' ') if x != '']
            matrix_nored = np.append(matrix_nored, [row], axis=0)

    remove(matrix_path)
    remove(matrix_nonredundant_path)

    if verbose:
        print('Removed %d redundant rows' % (matrix.shape[0] - matrix_nored.shape[0]))

    return matrix_nored[:, 1:]


def to_fractions(matrix, quasi_zero_correction=False, quasi_zero_tolerance=1e-13):
    if quasi_zero_correction:
        # Make almost zero values equal to zero
        matrix[(matrix < quasi_zero_tolerance) & (matrix > -quasi_zero_tolerance)] = Fraction(0, 1)

    fraction_matrix = matrix.astype('object')

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            # str() here makes Sympy use true fractions instead of the double-precision
            # floating point approximation
            fraction_matrix[row, col] = Fraction(str(matrix[row, col]))

    return fraction_matrix


def get_metabolite_adjacency(N):
    """
    Returns m by m adjacency matrix of metabolites, given
    stoichiometry matrix N. Diagonal is 0, not 1.
    :param N: stoichiometry matrix
    :return: m by m adjacency matrix
    """

    number_metabolites = N.shape[0]
    adjacency = np.zeros(shape=(number_metabolites, number_metabolites))

    for metabolite_index in range(number_metabolites):
        active_reactions = np.where(N[metabolite_index, :] != 0)[0]
        for reaction_index in active_reactions:
            adjacent_metabolites = np.where(N[:, reaction_index] != 0)[0]
            for adjacent in [i for i in adjacent_metabolites if i != metabolite_index]:
                adjacency[metabolite_index, adjacent] = 1
                adjacency[adjacent, metabolite_index] = 1

    return adjacency