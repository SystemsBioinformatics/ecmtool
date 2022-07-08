import os
import sys
from fractions import Fraction
from os import remove, devnull as os_devnull, system
from random import randint
from subprocess import check_call

import numpy as np
import psutil
from numpy.linalg import svd
from sympy import Matrix

from ecmtool.mpi_wrapper import get_process_rank


def unique(matrix):
    unique_set = list({tuple(row) for row in matrix if np.count_nonzero(row) > 0})
    return np.vstack(unique_set) if len(unique_set) else to_fractions(np.ndarray(shape=(0, matrix.shape[1])))


def find_unique_inds(matrix, verbose=False, tol=1e-9):
    n_rays = matrix.shape[0]
    n_nonunique = 0
    original_inds_remaining = np.arange(n_rays)
    unique_inds = []
    counter = 0
    while matrix.shape[0] > 0:
        row = matrix[0, :]
        unique_inds.append(original_inds_remaining[0])
        if verbose:
            if counter % 100 == 0:
                mp_print("Find unique rows has tested %d of %d (%f %%). Removed %d non-unique rows." %
                         (counter, n_rays, counter / n_rays * 100, n_nonunique))
        counter = counter + 1
        equal_rows = np.where(np.max(np.abs(matrix - row), axis=1) < tol)[0]
        if len(equal_rows):
            n_nonunique = n_nonunique + len(equal_rows) - 1
            matrix = np.delete(matrix, equal_rows, axis=0)
            original_inds_remaining = np.delete(original_inds_remaining, equal_rows)
        else:  # Something is wrong, at least the row itself should be equal to itself
            mp_print('Something is wrong in the unique_inds function!!')

    return unique_inds


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
    return psutil.virtual_memory().total / 1024 ** 3


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


def nullspace(N, symbolic=True, atol=1e-13, rtol=0):
    """
    Calculates the null space of given matrix N.
    Source: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    :param N: ndarray
            A should be at most 2-D.  A 1-D array with length k will be treated
            as a 2-D with shape (1, k)
    :param symbolic: set to False to compute nullspace numerically instead of symbolically
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
            np.transpose(np.asarray(nullspace_matrix, dtype='object'))) if nullspace_matrix \
            else np.ndarray(shape=(N.shape[0], 0))


def write_mplrs_matrix(textfile, matrix):
    """
    Writes ndarray to ine file for calculations with mplrs.
    :param textfile: path to output file
    :param matrix: ndarray
    :return: None
    """
    for line in matrix:
        textfile.write('0' + ' ')
        for val in line:
            textfile.write(str(val) + ' ')
        textfile.write('\n')
    

def write_header_no_linearity(textfile, inequality_matrix, d):
    """
    Writes header of ine file for calculations with mplrs if equality matrix is None.
    :param textfile: path to output file
    :param inequality_matrix: ndarray with inequalities
    :param d: dimensional parameter for ine file giving the width of the matrix
    :return: None
    """
    textfile.write('H-representation' + ' \n')
    textfile.write('begin' + ' \n')
    textfile.write(str(len(inequality_matrix)) + ' ' + str(d) +  ' ' + 'rational' + ' \n')
    
    
def write_header_with_linearity(textfile, linearity, d, m=None):
    """
    Writes header of ine file for calculations with mplrs.
    :param textfile: path to output file
    :param linearity: number of rows that are equations (mplrs parameter)
    :param d: dimensional parameter for ine file giving the width of the matrix (mplrs parameter)
    :param m: dimensional parameter for ine file giving the number of rows (mplrs parameter)
    :return: None
    """
    linearity_list = [*range(1,linearity+1,1)]
    textfile.write('H-representation' + ' \n')
    textfile.write('linearity' + ' ' + str(linearity) + ' ')
    for i in linearity_list:
        textfile.write(str(i) + ' ')
    textfile.write(' \n')
    textfile.write('begin' + ' \n')

    if m is None:
        textfile.write(str(linearity) + ' ' + str(d) +  ' ' + 'rational' + ' \n')
    else:
        textfile.write(str(m) + ' ' + str(d) +  ' ' + 'rational' + ' \n')
          
    
def write_mplrs_input(equality_matrix, inequality_matrix, mplrs_path, verbose=False):
    """

    :param equality_matrix: ndarray with equalities
    :param inequality_matrix: ndarray with inequalities
    :param mplrs_path: absolute path to mplrs binary
    :param verbose: print status messages during enumeration
    :return: d dimensional parameter for ine file giving the width of the matrix
    """
    if inequality_matrix is not None:
        inequality_matrix = inequality_matrix.tolist()

    if equality_matrix is not None:
        equality_matrix = equality_matrix.tolist()

    textfile = open(mplrs_path, "w")

    if equality_matrix is None:
        d = len(inequality_matrix[0] ) + 1
        write_header_no_linearity(textfile, inequality_matrix, d, m=None)
        if verbose:
            print('Writing inequalities to file')
        write_mplrs_matrix(textfile, inequality_matrix)

    elif inequality_matrix is None:
        linearity = len(equality_matrix)
        d = len(equality_matrix[0] ) + 1
        write_header_with_linearity(textfile, linearity, d, m=None)
        if verbose:
            print('Writing equalities to file')
        write_mplrs_matrix(textfile, equality_matrix)

    else:
        linearity = len(equality_matrix)
        d = len(equality_matrix[0] ) + 1
        m = linearity + len(inequality_matrix)
        write_header_with_linearity(textfile, linearity, d, m)
        if verbose:
            print('Writing equalities to file')
        write_mplrs_matrix(textfile, equality_matrix)
        if verbose:
            print('Writing inequalities to file')
        write_mplrs_matrix(textfile, inequality_matrix)

    textfile.write('end' + ' \n')
    textfile.close()
    return d


def parse_mplrs_output(mplrs_output_path, d, verbose=False):
    """
    Parses mplrs output file. Removes header and footer, as well as origin vertex (first column in matrix of outputfile)
    :param mplrs_output_path: absolute path to mplrs binary
    :param d: dimensional parameter for ine file giving the width of the matrix
    :param verbose: print status messages during enumeration
    :return: computed rays as ndarray
    """
    if verbose:
        print('Parsing computed rays')
        
    with open(mplrs_output_path, 'r') as file:
        lines = file.readlines()
        rays_vertices = np.ndarray(shape=(0, d))

        if len(lines) > 0:
            begin = lines.index("begin\n")
            start = begin + 3
            stop = lines.index("end\n")
            end = len(lines) - stop
            del lines[-end:]
            del lines[:start]
            
            number_lines = len(lines)
            rays_vertices = np.repeat(np.repeat(to_fractions(np.zeros(shape=(1, 1))), d, axis=1), number_lines, axis=0)

            for row, line in enumerate(lines):
                for column, value in enumerate(line.replace('\n', '').split()):
                    if value != '0':
                        rays_vertices[row, column] = Fraction(str(value))
                        
    rays = rays_vertices[:,1:]
    
    return rays


def get_extreme_rays_mplrs(equality_matrix, inequality_matrix, processes, rand, path2mplrs, verbose=False):
    """
    Calculates extreme rays using mplrs. Includes generation of input file, calculations and parsing of output file.
    :param equality_matrix: ndarray containing all equalities
    :param inequality_matrix: ndarray containing all inequalities
    :param processes: integer value giving the number of processes
    :param rand: random integer value for generating tmp file names
    :param path2mplrs: absolute path to mplrs binary
    :param verbose: print status messages during enumeration
    :return: ndarray with computed rays
    """

    mplrs_input_path = relative_path('tmp' + os.sep + 'mplrs_%d.ine' % rand)
    mplrs_redund_path = relative_path('tmp' + os.sep + 'redund_%d.ine' % rand)
    mplrs_output_path = relative_path('tmp' + os.sep + 'mplrs_%d.out' % rand)

    # Write mplrs input files to disk
    width_matrix = write_mplrs_input(equality_matrix, inequality_matrix, mplrs_input_path, verbose=False)
    
    if path2mplrs is None:
        if verbose:
            print('Running mplrs redund')
        check_call((('mpirun -np 3 mplrs -redund %s' % mplrs_input_path) + (' %s' % mplrs_redund_path)),shell=True)
        
        if verbose:
            print('Running mplrs with %s processes' % processes)
        check_call((('mpirun -np %s' % processes) + (' mplrs %s' % mplrs_redund_path) + (' %s' % mplrs_output_path)),shell=True)
 
    else:
        if verbose:
            print('Running mplrs redund')
        check_call((('mpirun -np 3 %s' % path2mplrs) + (' -redund %s' % mplrs_input_path) + (' %s' % mplrs_redund_path)),shell=True)
        
        if verbose:
            print('Running mplrs with %s processes' % processes)
        check_call((('mpirun -np %s' % processes) + (' %s' % path2mplrs) + (' %s' % mplrs_redund_path) + (' %s' % mplrs_output_path)),shell=True)

    # Parse resulting extreme rays
    rays = parse_mplrs_output(mplrs_output_path, width_matrix, verbose=False)

    if verbose:
        print('Done parsing rays')

    # Clean up the files created above
    remove(mplrs_input_path)
    remove(mplrs_redund_path)
    remove(mplrs_output_path)

    return rays

    
def get_extreme_rays_polco(equality_matrix, inequality_matrix, rand, jvm_mem, processes, symbolic=True, verbose=False):
    """
    Calculates extreme rays using polco. Includes generation of input file, calculations and parsing of output file.
    :param equality_matrix: equality_matrix: ndarray containing all equalities
    :param inequality_matrix: inequality_matrix: ndarray containing all inequalities
    :param rand: random integer value for generating tmp file names
    :param jvm_mem: tuple of integer giving the minimum and maximum number of java VM memory in GB
    :param processes: integer value giving the number of processes
    :param symbolic: set to False to compute nullspace numerically instead of symbolically
    :param verbose: print status messages during enumeration
    :return: ndarray with computed rays
    """
    # Write equalities system to disk as space separated file
    if verbose:
        print('Writing equalities to file')
    if equality_matrix is not None:
        with open_relative('tmp' + os.sep + 'eq_%d.txt' % rand, 'w') as file:
            for row in range(equality_matrix.shape[0]):
                file.write(' '.join([str(val) for val in equality_matrix[row, :]]) + '\n')

    # Write inequalities system to disk as space separated file
    if verbose:
        print('Writing inequalities to file')
    with open_relative('tmp' + os.sep + 'iq_%d.txt' % rand, 'w') as file:
        for row in range(inequality_matrix.shape[0]):
            file.write(' '.join([str(val) for val in inequality_matrix[row, :]]) + '\n')

    # Run external extreme ray enumeration tool
    if jvm_mem == None:
        min_mem, max_mem = get_min_max_java_memory()
    else:
        min_mem, max_mem = jvm_mem   
    
    if verbose:
        print('Running polco (%d-%d GiB java VM memory)' % (min_mem, max_mem), 'with "-maxthreads" %s' % processes)

    equality_path = relative_path('tmp' + os.sep + 'eq_%d.txt' % rand)
    inequality_path = relative_path('tmp' + os.sep + 'iq_%d.txt' % rand)
    generators_path = relative_path('tmp' + os.sep + 'generators_%d.txt' % rand)
    with open(os_devnull, 'w') as devnull:
        polco_path = relative_path('polco' + os.sep + 'polco.jar')
        check_call(('java -Xms%dg -Xmx%dg ' % (min_mem, max_mem) +
                    '-jar %s -kind text -sortinput AbsLexMin ' % polco_path +
                    '-maxthreads %s ' % processes +
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
            rays = np.repeat(np.repeat(to_fractions(np.zeros(shape=(1, 1))), number_entries, axis=1), number_lines,
                             axis=0)

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


def get_extreme_rays(equality_matrix=None, inequality_matrix=None, verbose=False, polco=False, processes=None, jvm_mem=None, path2mplrs=None):
    """
    Calculates extreme rays using either mplrs or polco.
    :param equality_matrix: equality_matrix: ndarray containing all equalities
    :param inequality_matrix: inequality_matrix: ndarray containing all inequalities
    :param verbose: print status messages during enumeration
    :param polco: set to True to make computation with polco
    :param processes: integer value giving the number of processes
    :param jvm_mem: tuple of integer giving the minimum and maximum number of java VM memory in GB
    :param path2mplrs: absolute path to mplrs binary
    :return: ndarray with computed rays
    """
    if not os.path.isdir(relative_path('tmp')):
        os.makedirs(relative_path('tmp'))

    rand = randint(1, 10 ** 6)

    if inequality_matrix is not None and inequality_matrix.shape[0] == 0:
        inequality_matrix = None

    if equality_matrix is not None and equality_matrix.shape[0] == 0:
        equality_matrix = None

    if inequality_matrix is None:
        if equality_matrix is not None:
            inequality_matrix = np.zeros(shape=(1, equality_matrix.shape[1]))
        else:
            raise Exception('No equality or inequality argument given')

    if polco is True:
        if verbose:
            print('Starting enumeration with polco')
        rays = get_extreme_rays_polco(equality_matrix, inequality_matrix, rand, jvm_mem, processes, symbolic=True, verbose=verbose)
    else:
        if verbose:
            print('Starting enumeration with mplrs')
        rays = get_extreme_rays_mplrs(equality_matrix, inequality_matrix, processes, rand, path2mplrs, verbose=verbose)
        

    return rays


def binary_exists(binary_file):
    return any(
        os.access(os.path.join(path, binary_file), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )


def get_redund_binary():
    if sys.platform.startswith('linux'):
        if not binary_exists('redund'):
            raise EnvironmentError(
                'Executable "redund" was not found in your path. Please install package lrslib (e.g. apt install lrslib)')
        return 'redund'
    elif sys.platform.startswith('win32'):
        return relative_path('redund\\redund_win.exe')
    elif sys.platform.startswith('darwin'):
        return relative_path('redund/redund_mac')
    else:
        raise OSError('Unsupported operating system platform: %s' % sys.platform)


def redund(matrix, verbose=False):
    if not os.path.isdir(relative_path('tmp')):
        os.makedirs(relative_path('tmp'))
    rank = str(get_process_rank())
    matrix = to_fractions(matrix)
    binary = get_redund_binary()
    matrix_path = relative_path('tmp' + os.sep + 'matrix' + rank + '.ine')
    matrix_nonredundant_path = relative_path('tmp' + os.sep + 'matrix_nored' + rank + '.ine')

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

    if not os.path.exists(matrix_nonredundant_path):
        raise ValueError('An error occurred during removal of redundant vectors from an input matrix: '
                         'redund did not write an output file after being presented input file "%s". \r\n\r\n'
                         'Please check if your input matrix contains erroneous data, and let us know via https://github.com/SystemsBioinformatics/ecmtool/issues '
                         'if you think the input matrix seems fine. It helps if you attach the matrix file mentioned above when creating an issue.')

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


def mp_print(*args, **kwargs):
    """
    Multiprocessing wrapper for print().
    Prints the given arguments, but only on process 0 unless
    named argument PRINT_IF_RANK_NONZERO is set to true.
    :return:
    """
    if get_process_rank() == 0:
        print(*args)
    elif 'PRINT_IF_RANK_NONZERO' in kwargs and kwargs['PRINT_IF_RANK_NONZERO']:
        print(*args)


def unsplit_metabolites(R, network):
    metabolite_ids = [metab.id for metab in network.metabolites]
    res = []
    ids = []

    processed = {}
    for i in range(R.shape[0]):
        metabolite = metabolite_ids[i].replace("_virtin", "").replace("_virtout", "")
        if metabolite in processed:
            row = processed[metabolite]
            res[row] += R[i, :]
        else:
            if metabolite[-4:] == '_int':
                if np.sum(np.abs(R[i, :])) == 0:
                    continue
                else:
                    mp_print("Metabolite " + metabolite + " was made internal, but now is nonzero in ECMs.")
            res.append(R[i, :].tolist())
            processed[metabolite] = len(res) - 1
            ids.append(metabolite)

    # remove all-zero rays
    res = np.asarray(res)
    res = res[:, [sum(abs(res)) != 0][0]]

    return res, ids


def print_ecms_direct(R, metabolite_ids):
    obj_id = -1
    if "objective" in metabolite_ids:
        obj_id = metabolite_ids.index("objective")
    elif "objective_virtout" in metabolite_ids:
        obj_id = metabolite_ids.index("objective_virtout")

    mp_print("\n--%d ECMs found--\n" % R.shape[1])
    for i in range(R.shape[1]):
        mp_print("ECM #%d:" % (i + 1))
        if np.max(R[:,
                  i]) > 1e100:  # If numbers become too large, they can't be printed, therefore we make them smaller first
            ecm = np.array(R[:, i] / np.max(R[:, i]), dtype='float')
        else:
            ecm = np.array(R[:, i], dtype='float')

        div = 1
        if obj_id != -1 and R[obj_id][i] != 0:
            div = ecm[obj_id]
        for j in range(R.shape[0]):
            if ecm[j] != 0:
                mp_print("%s\t\t->\t%.4f" % (metabolite_ids[j].replace("_in", "").replace("_out", ""), ecm[j] / div))
        mp_print("")


def normalize_columns_slower(R, verbose=False):  # This was the original function, but seems slower and further equivalent to new function below
    result = np.zeros(R.shape)
    number_rays = R.shape[1]
    for i in range(result.shape[1]):
        if verbose:
            if i % 10000 == 0:
                mp_print("Normalize columns is on ray %d of %d (%f %%)" %
                         (i, number_rays, i / number_rays * 100), PRINT_IF_RANK_NONZERO=True)
        largest_number = np.max(np.abs(R[:,i]))
        if largest_number > 1e100:  # If numbers are very large, converting to float might give issues, therefore we first divide by another int
            part_normalized_column = np.array(R[:, i] / largest_number, dtype='float')
            result[:, i] = part_normalized_column / np.linalg.norm(part_normalized_column, ord=1)
        else:
            norm_column = np.linalg.norm(np.array(R[:, i], dtype='float'), ord=1)
            if norm_column != 0:
                result[:, i] = np.array(R[:, i], dtype='float') / norm_column
    return result


def normalize_columns(R, verbose=False):
    result = R
    largest_number = max(np.max(R), -np.min(R))
    if largest_number > 1e100:
        result = result / largest_number # If numbers are very large, converting to float might give issues, therefore we first divide by another int
    norms = np.linalg.norm(result, axis=0, ord=1)
    norms[np.where(norms==0)[0]] = 1
    result = np.array(np.divide(result, norms), dtype='float')
    return result


def find_remaining_rows(first_mat, second_mat, tol=1e-12, verbose=False):
    """Checks which rows (indices) of second_mat are still in first_mat"""
    remaining_inds = []
    number_rays = first_mat.shape[0]
    for ind, row in enumerate(first_mat):
        if verbose:
            if ind % 10000 == 0:
                mp_print("Find remaining rows is on row %d of %d (%f %%)" %
                         (ind, number_rays, ind / number_rays * 100))
        sec_ind = np.where(np.max(np.abs(second_mat - row), axis=1) < tol)[0]
        # for sec_ind, sec_row in enumerate(second_mat):
        #    if np.max(np.abs(row - sec_row)) < tol:
        #        remaining_inds.append(ind)
        #        continue
        if len(sec_ind):
            remaining_inds.append(sec_ind[0])
        else:
            mp_print('Warning: There are rows in the first matrix that are not in the second matrix')

    return remaining_inds


def normalize_columns_fraction(R, vectorized=False, verbose=True):
    if not vectorized:
        number_rays = R.shape[1]
        for i in range(number_rays):
            if verbose:
                if i % 10000 == 0:
                    mp_print("Normalize columns is on ray %d of %d (%f %%)" %
                             (i, number_rays, i / number_rays * 100), PRINT_IF_RANK_NONZERO=True)
            norm_column = np.sum(np.abs(np.array(R[:, i])))
            if norm_column!=0:
                R[:, i] = np.array(R[:, i]) / norm_column
    else:
        R = R / np.sum(np.abs(R), axis=0)
    return R
