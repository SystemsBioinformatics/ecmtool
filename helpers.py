import psutil
from fractions import Fraction, gcd
from subprocess import check_call, STDOUT, PIPE
from os import remove, devnull as os_devnull

import cbmpy
import matlab.engine

import cdd
import numpy as np
import libsbml as sbml
from random import randint

from numpy.linalg import svd
from sympy import Matrix

from network import Network, Reaction, Metabolite

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


def get_sbml_model(path):
    doc = sbml.readSBMLFromFile(path)
    model = doc.getModel()
    model.__keep_doc_alive__ = doc
    return model


def extract_sbml_stoichiometry(path, add_objective=True, skip_external_reactions=True, determine_inputs_outputs=False, external_compartment='e'):
    """
    Parses an SBML file containing a metabolic network, and returns a Network instance
    with the metabolites, reactions, and stoichiometry initialised. By default will look
    for an SBML v3 FBC objective function, and skip reactions that contain '_EX_' in their ID.
    :param path: string absolute or relative path to the .sbml file
    :param add_objective: Look for SBML v3 FBC objective definition
    :param skip_external_reactions: Ignore external reactions, as identified by '_EX_' in their ID
    :return: Network
    """
    model = get_sbml_model(path)
    species = list(model.species)
    species_index = {item.id: index for index, item in enumerate(species)}
    reactions = model.reactions
    objective_reaction_column = None


    # TODO: parse stoichiometry, reactions, and metabolites using CBMPy too
    cbmpy_model = cbmpy.readSBML3FBC(path)
    pairs = cbmpy.CBTools.findDeadEndReactions(cbmpy_model)
    external_metabolites, external_reactions = zip(*pairs) if len(pairs) else (zip(*cbmpy.CBTools.findDeadEndMetabolites(cbmpy_model))[0], [])

    # Catch any metabolites that were not recognised automatically, but are likely external
    external_metabolites = list(external_metabolites) + [item.id for item in species if item.compartment == external_compartment]

    network = Network()
    network.metabolites = [Metabolite(item.id, item.name, item.compartment, item.id in external_metabolites) for item in species]

    if add_objective:
        plugin = model.getPlugin('fbc')
        objective_name = plugin.getObjective(0).flux_objectives[0].reaction

    if skip_external_reactions:
        reactions = [reaction for reaction in reactions if reaction.id not in external_reactions]

    if determine_inputs_outputs:
        for metabolite in [network.metabolites[index] for index in network.external_metabolite_indices()]:
            index = external_metabolites.index(metabolite.id)
            if index >= len(external_reactions):
                print('Warning: missing exchange reaction for metabolite %s. Skipping marking this metabolite as input or output.' % metabolite.id)
                continue

            reaction_id = external_reactions[index]
            reaction = cbmpy_model.getReaction(reaction_id)
            lowerBound, upperBound, _ = cbmpy_model.getFluxBoundsByReactionID(reaction_id)
            stoichiometries = reaction.getStoichiometry()
            stoichiometry = [stoich[0] for stoich in stoichiometries if stoich[1] == metabolite.id][0]

            if reaction.reversible:
                # Check if the reaction is truly bidirectional
                if lowerBound.value == 0 or upperBound.value == 0:
                    reaction.reversible = False
                else:
                    # Reversible reactions are both inputs and outputs, so don't mark as either
                    continue

                if lowerBound.value > upperBound.value:
                    # Direction of model is inverted (substrates are products and vice versa. This happens sometimes,
                    # e.g. https://github.com/SBRG/bigg_models/issues/324
                    print('Swapping direction of reversible reaction %s that can only run in reverse direction.' % reaction_id)
                    stoichiometry *= -1
                    for met in model.getReaction(reaction_id).reactants:
                        met.setStoichiometry(-met.getStoichiometry())
                    for met in model.getReaction(reaction_id).products:
                        met.setStoichiometry(-met.getStoichiometry())

            metabolite.direction = 'input' if stoichiometry >= 0 else 'output'

    # Build stoichiometry matrix N
    N = np.zeros(shape=(len(species), len(reactions)), dtype='object')
    for column, reaction in enumerate(reactions):
        network.reactions.append(Reaction(reaction.id, reaction.name, reaction.reversible))

        for metabolite in reaction.reactants:
            row = species_index[metabolite.species]
            N[row, column] = Fraction(str(-metabolite.stoichiometry))
        for metabolite in reaction.products:
            row = species_index[metabolite.species]
            N[row, column] = Fraction(str(metabolite.stoichiometry))

        if add_objective and reaction.id == objective_name:
            objective_reaction_column = column

    # Add objective metabolite from objective reaction
    if add_objective and objective_reaction_column:
        network.metabolites.append(Metabolite('objective', 'Virtual objective metabolite', 'e', is_external=True, direction='output'))
        N = np.append(N, to_fractions(np.zeros(shape=(1, N.shape[1]))), axis=0)
        N[-1, objective_reaction_column] = 1

    network.N = N

    return network


def add_debug_tags(network, reactions=[]):
    if len(reactions) == 0:
        reactions = range(len(network.reactions))

    for reaction in reactions:
        network.metabolites.append(Metabolite('virtual_tag_%s' % network.reactions[reaction].id,
                                              'Virtual tag for %s' % network.reactions[reaction].id,
                                              compartment='e', is_external=True,
                                              direction='both' if network.reactions[reaction].reversible else 'output'))
    network.N = np.append(network.N, np.identity(len(network.reactions))[reactions, :], axis=0)


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