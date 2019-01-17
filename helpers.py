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


def nullspace_polco(N, verbose=False):
    return np.transpose(np.asarray(list(get_extreme_rays(N, None, verbose=verbose)), dtype='object'))


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
            inequality_matrix = np.identity(equality_matrix.shape[1])
        else:
            raise Exception('No equality or inequality argument given')

    # Write equalities system to disk as space separated file
    if verbose:
        print('Writing equalities to file')
    if equality_matrix is not None:
        with open('tmp/eq_%d.txt' % rand, 'w') as file:
            for row in range(equality_matrix.shape[0]):
                file.write(' '.join([str(val) for val in equality_matrix[row, :]]) + '\r\n')

    # Write inequalities system to disk as space separated file
    if verbose:
        print('Writing inequalities to file')
    with open('tmp/iq_%d.txt' % rand, 'w') as file:
        for row in range(inequality_matrix.shape[0]):
            file.write(' '.join([str(val) for val in inequality_matrix[row, :]]) + '\r\n')

    # Run external extreme ray enumeration tool
    min_mem, max_mem = get_min_max_java_memory()
    if verbose:
        print('Running polco (%d-%d GiB java VM memory)' % (min_mem, max_mem))
    with open(os_devnull, 'w') as devnull:
        check_call(('java -Xms%dg -Xmx%dg ' % (min_mem, max_mem) +
                    '-jar polco/polco.jar -kind text -sortinput AbsLexMin ' +
                    '-arithmetic %s ' % (' '.join(['fractional' if symbolic else 'double'] * 3)) +
                    '-zero %s ' % (' '.join(['NaN' if symbolic else '1e-10'] * 3)) +
                    ('' if equality_matrix is None else '-eq tmp/eq_%d.txt ' % (rand)) +
                    ('' if inequality_matrix is None else '-iq tmp/iq_%d.txt ' % (rand)) +
                    '-out text tmp/generators_%d.txt' % rand).split(' '),
            stdout=(devnull if not verbose else None), stderr=(devnull if not verbose else None))

    # Read resulting extreme rays
    if verbose:
        print('Parsing computed rays')
    with open('tmp/generators_%d.txt' % rand, 'r') as file:
        lines = file.readlines()

        for index, line in enumerate(lines):
            result = []
            for value in line.replace('\n', '').split('\t'):
                result.append(Fraction(str(value)))
            yield np.transpose(result)

    # Clean up the files created above
    if equality_matrix is not None:
        remove('tmp/eq_%d.txt' % rand)

    remove('tmp/iq_%d.txt' % rand)
    remove('tmp/generators_%d.txt' % rand)

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
        return 'redund/redund_win.exe'
    elif sys.platform.startswith('darwin'):
        return 'redund/redund_mac'
    else:
        raise OSError('Unsupported operating system platform: %s' % sys.platform)


def redund(matrix, verbose=False):
    matrix = to_fractions(matrix)
    binary = get_redund_binary()

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

    system('%s tmp/matrix.ine > tmp/matrix_nored.ine' % binary)

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