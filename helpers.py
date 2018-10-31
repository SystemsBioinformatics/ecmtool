from fractions import Fraction
from subprocess import check_call, STDOUT, PIPE
from os import remove, devnull as os_devnull

import cdd
import numpy as np
import libsbml as sbml
from random import randint
from numpy.linalg import svd
from sympy import Matrix

from network import Network, Reaction, Metabolite


def get_P_M(metabolites, metabolic_reactions, enzyme_reactions):
    P = np.zeros(shape=(len(metabolites), len(metabolic_reactions)))
    for col, reaction in enumerate(metabolic_reactions):
        for metabolite in reaction.keys():
            row = metabolites.index(metabolite)
            P[row, col] = reaction[metabolite]

    M = np.zeros(shape=(len(metabolites), len(enzyme_reactions)))
    for col, reaction in enumerate(enzyme_reactions):
        for metabolite in reaction.keys():
            row = metabolites.index(metabolite)
            M[row, col] = -reaction[metabolite]

    return P, M


def get_net_volumes(metabolites, metabolic_reactions, metabolite_molar_volumes, enzyme_reactions, enzyme_molar_volumes,
                    P, M):
    a = [np.sum([metabolite_molar_volumes[k] * P[k, j] for k in range(len(metabolites))])
         for j in range(len(metabolic_reactions))]

    b = [
        enzyme_molar_volumes[j] - np.sum([metabolite_molar_volumes[k] * M[k, j] for k in range(len(metabolites))])
        for j in range(len(enzyme_reactions))]

    return a, b


def build_A_matrix(metabolites, metabolic_reactions, metabolic_concentrations, P, M, net_volume_a, net_volume_b,
                   rate_functions, growth_rate):
    A = np.zeros(shape=(len(metabolites) + 1, len(metabolic_reactions) + 1))
    for k, metabolite in enumerate(metabolites):
        for j in range(len(metabolic_reactions)):
            A[k, j] = (metabolic_concentrations[metabolite] * net_volume_a[k] - P[k, j]) * \
                      (rate_functions[j](metabolic_concentrations) / growth_rate) + \
                      (metabolic_concentrations[metabolite] * net_volume_b[j]) + M[k, j]
    for k, metabolite in enumerate(metabolites):
        r = len(metabolic_reactions)
        # Note that we have zero-based indexing here, so we use r instead of r+1 like in the paper
        A[k, r] = metabolic_concentrations[metabolite] * net_volume_b[r] + M[k, r]

    # Last row, last column should be set to 1
    A[-1, -1] = 1

    # During development, remove last row such that A*x = 0 instead of A*x = L*e_(m+1)*mu
    A = A[:-1, ]

    return A


def build_reduced_A_matrix(k_list, j_list, metabolic_concentrations, P, M, net_volume_a, net_volume_b,
                           rate_functions, growth_rate):
    A = np.zeros(shape=(len(k_list) + 1, len(j_list) + 1))
    metabolite_names = list(metabolic_concentrations.keys())

    for row, k in enumerate(k_list):
        concentration = metabolic_concentrations[metabolite_names[k]]
        for col, j in enumerate(j_list):
            A[row, col] = (concentration * net_volume_a[k] - P[k, j]) * \
                          (rate_functions[j](metabolic_concentrations) / growth_rate) + \
                          (concentration * net_volume_b[j]) + M[k, j]

        r = len(j_list)
        # Note that we have zero-based indexing here, so we use r instead of r+1 like in the paper
        A[k, r] = concentration * net_volume_b[-1] + M[k, -1]

    # Last row, last column should be set to growth_rate
    A[-1, -1] = growth_rate

    return A


def normalise_betas(result):
    amount_betas = len(result) - 2
    for i in range(amount_betas):
        # Divide all Beta_i by their Beta_r+1 (ribosome synthesis rate)
        result[2 + i] /= result[-1]

        # Multiply normalised Beta_r+1 (ribosome synthesis rate) by the growth rate (sets Beta_r+1 = mu)
        result[-1] *= result[0]
    return result


def get_extreme_rays_cdd(inequality_matrix):
    mat = cdd.Matrix(np.append(np.zeros(shape=(inequality_matrix.shape[0], 1)), inequality_matrix, axis=1), number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    gen = poly.get_generators()
    return np.asarray(gen)[:, 1:]



def get_extreme_rays(equality_matrix=None, inequality_matrix=None, fractional=True, verbose=False):
    rand = randint(1, 10 ** 6)

    if inequality_matrix is None:
        if equality_matrix is not None:
            inequality_matrix = np.identity(equality_matrix.shape[1])
        else:
            raise Exception('No equality or inequality argument given')

    # Write equalities system to disk as space separated file
    if equality_matrix is not None:
        with open('tmp/egm_eq_%d.txt' % rand, 'w') as file:
            for row in range(equality_matrix.shape[0]):
                file.write(' '.join([str(val) for val in equality_matrix[row, :]]) + '\r\n')

    # Write inequalities system to disk as space separated file
    with open('tmp/egm_iq_%d.txt' % rand, 'w') as file:
        for row in range(inequality_matrix.shape[0]):
            file.write(' '.join([str(val) for val in inequality_matrix[row, :]]) + '\r\n')

    # Run external extreme ray enumeration tool
    with open(os_devnull, 'w') as devnull:
        check_call(('java -Xms1g -Xmx7g -jar polco.jar -kind text ' +
                    '-arithmetic %s ' % (' '.join(['fractional' if fractional else 'double'] * 3)) +
                    ('' if equality_matrix is None else '-eq tmp/egm_eq_%d.txt ' % (rand)) +
                    '-iq tmp/egm_iq_%d.txt -out text tmp/generators_%d.txt' % (rand, rand)).split(' '),
            stdout=(devnull if not verbose else None), stderr=(devnull if not verbose else None))

    # Read resulting extreme rays
    with open('tmp/generators_%d.txt' % rand, 'r') as file:
        lines = file.readlines()

        for index, line in enumerate(lines):
            result = []
            for value in line.replace('\n', '').split('\t'):
                result.append(Fraction(str(value)))
            yield result

    # Clean up the files created above
    if equality_matrix is not None:
        remove('tmp/egm_eq_%d.txt' % rand)

    remove('tmp/egm_iq_%d.txt' % rand)
    remove('tmp/generators_%d.txt' % rand)


def add_rates(result, rate_functions, metabolite_concentrations, net_volume_a, net_volume_b, M):
    amount_reactions = len(rate_functions)
    growth_rate = result[0]
    betas = result[2:-1]
    row = []

    # Defined between eq. 25 and eq. 26
    ribosome_concentration = (growth_rate / ((net_volume_b[-1] * growth_rate) +
                                             (np.sum([
                                                 (net_volume_a[j] * (
                                                     rate_functions[j](metabolite_concentrations) / growth_rate) +
                                                  net_volume_b[j]) * betas[j]
                                                 for j in range(amount_reactions)]))
                                             )
                              )

    enzyme_concentrations = [ribosome_concentration * (betas[j] / growth_rate) for j in range(amount_reactions)]
    row = np.append(row, ribosome_concentration)

    row = np.append(row, [enzyme_concentrations[i] for i in range(amount_reactions)])

    # Calculate Vi
    row = np.append(row, [enzyme_concentrations[i] * rate_functions[i](metabolite_concentrations) for i in
                          range(amount_reactions)])

    # Calculate Wi
    row = np.append(row, [ribosome_concentration * betas[i] for i in range(amount_reactions)])
    row = np.append(row, [ribosome_concentration * growth_rate])

    Mw = np.dot(M, row[-(amount_reactions + 1):])
    row = np.append(row, Mw)

    result = np.append(result, row)
    return result


def get_used_rows_columns(A_matrix, result):
    betas = result[2:-1]

    active_columns = [index for index, beta in enumerate(betas) if beta > 0]
    active_rows = [row for row in range(A_matrix.shape[0]) if np.sum(A_matrix[row, active_columns]) != 0]

    return active_columns, active_rows


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
        N = np.asarray(N, dtype='float64')
        u, s, vh = svd(N)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj()
        return ns
    else:
        nullspace_vectors = Matrix(N).nullspace()

        # Add nullspace vectors to a nullspace matrix as row vectors
        # Must be a sympy Matrix so we can do rref()
        nullspace_matrix = nullspace_vectors[0].T if len(nullspace_vectors) else None
        for i in range(1, len(nullspace_vectors)):
            nullspace_matrix = nullspace_matrix.row_insert(-1, nullspace_vectors[i].T)

        return to_fractions(
            np.asarray(nullspace_matrix.rref()[0], dtype='object')) if nullspace_matrix \
            else np.ndarray(shape=(0, N.shape[0]))


def get_sbml_model(path):
    doc = sbml.readSBMLFromFile(path)
    model = doc.getModel()
    return model


def extract_sbml_stoichiometry(path, add_objective=True, skip_external_reactions=True):
    """
    Parses an SBML file containing a metabolic network, and returns a Network instance
    with the metabolites, reactions, and stoichiometry initialised. By default will look
    for an SBML v3 FBC objective function, and skip reactions that contain '_EX_' in their ID.
    :param path: string absolute or relative path to the .sbml file
    :param add_objective: Look for SBML v3 FBC objective definition
    :param skip_external_reactions: Ignore external reactions, as identified by '_EX_' in their ID
    :return: Network
    """
    doc = sbml.readSBMLFromFile(path)
    model = doc.getModel()
    species = list(model.species)
    species_index = {item.id: index for index, item in enumerate(species)}
    reactions = model.reactions
    objective_reaction_column = None

    network = Network()
    network.metabolites = [Metabolite(item.id, item.name, item.compartment) for item in species]

    if add_objective:
        plugin = model.getPlugin('fbc')
        objective_name = plugin.getObjective(0).flux_objectives[0].reaction

    if skip_external_reactions:
        reactions = [reaction for reaction in reactions if '_EX_' not in reaction.id]

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

    if add_objective and objective_reaction_column:
        network.metabolites.append(Metabolite('objective', 'Virtual objective metabolite', 'e'))
        N = np.append(N, np.zeros(shape=(1, N.shape[1])), axis=0)
        N[-1, objective_reaction_column] = 1

    network.N = N

    return network


def add_debug_tags(network):
    for reaction in network.reactions:
        network.metabolites.append(Metabolite('virtual_tag_%s' % reaction.id, 'Virtual tag for %s' % reaction.id, compartment='e'))
    network.N = np.append(network.N, np.identity(len(network.reactions)), axis=0)


def to_fractions(matrix, quasi_zero_correction=True, quasi_zero_tolerance=1e-13):
    if quasi_zero_correction:
        # Make almost zero values equal to zero
        matrix[(matrix < quasi_zero_tolerance) & (matrix > -quasi_zero_correction)] = 0

    fraction_matrix = matrix.astype('object')

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            fraction_matrix[row, col] = Fraction(str(matrix[row, col]))

    return fraction_matrix