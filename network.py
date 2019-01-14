from cdd import Fraction
from os import system, remove

from scipy.optimize import linprog
import numpy as np


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


def redund(matrix, verbose=False):
    matrix = to_fractions(matrix)

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

    system('scripts/redund tmp/matrix.ine > tmp/matrix_nored.ine')

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


def clementine_equality_compression(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                                    verbose=True):
    """
    Calculates the conversion cone using Superior Clementine Equality Intersection (all rights reserved).
    Follows the general Double Description method by Motzkin, using G as initial basis and intersecting
    hyperplanes of internal metabolites = 0.
    :param N:
    :param external_metabolites:
    :param reversible_reactions:
    :param input_metabolites:
    :param output_metabolites:
    :return:
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]
    internal_metabolites = np.setdiff1d(range(amount_metabolites), external_metabolites)

    identity = np.identity(amount_metabolites)
    equalities = [identity[:, index] for index in internal_metabolites]

    # Compose G of the columns of N
    G = np.transpose(N)

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_reactions:
            G = np.append(G, [-G[reaction_index, :]], axis=0)

    # For each internal metabolite, intersect the intermediary cone with an equality to 0 for that metabolite
    for index, internal_metabolite in enumerate(internal_metabolites):
        if verbose:
            print('Iteration %d/%d' % (index, len(internal_metabolites)))

        # Find conversions that use this metabolite
        active_conversions = np.asarray([conversion_index for conversion_index in range(G.shape[0])
                              if G[conversion_index, internal_metabolite] != 0])

        # Skip internal metabolites that aren't used anywhere
        if len(active_conversions) == 0:
            if verbose:
                print('Skipping internal metabolite #%d, since it is not used by any reaction\n' % internal_metabolite)
            continue

        # Skip internal metabolites that are used too often (>= busy_threshold)
        busy_threshold = 10
        if len(active_conversions) >= busy_threshold:
            if verbose:
                print('Skipping internal metabolite #%d, since it is used by too many reactions\n' % internal_metabolite)
            continue

        # Project conversions that use this metabolite onto the hyperplane internal_metabolite = 0
        projections = np.dot(G[active_conversions, :], equalities[index])
        positive = active_conversions[np.argwhere(projections > 0)[:, 0]]
        negative = active_conversions[np.argwhere(projections < 0)[:, 0]]
        candidates = np.ndarray(shape=(0, amount_metabolites))

        if verbose:
            print('Adding %d candidates' % (len(positive) * len(negative)))

        # Make convex combinations of all pairs (positive, negative) such that their internal_metabolite = 0
        for pos in positive:
            for neg in negative:
                candidate = np.add(G[pos, :], G[neg, :] * (G[pos, internal_metabolite] / -G[neg, internal_metabolite]))
                candidates = np.append(candidates, [candidate], axis=0)

        # Keep only rays that satisfy internal_metabolite = 0
        keep = np.setdiff1d(range(G.shape[0]), np.append(positive, negative, axis=0))
        if verbose:
            print('Removing %d rays\n' % (G.shape[0] - len(keep)))
        G = G[keep, :]
        G = np.append(G, candidates, axis=0)
        # G = drop_nonextreme(G, get_zero_set(G, equalities), verbose=verbose)
        G = redund(G, verbose=verbose)

    return G

class Reaction:
    id = ''
    name = ''
    reversible = False

    def __init__(self, id, name, reversible):
        self.id, self.name, self.reversible = id, name, reversible


class Metabolite:
    id = ''
    name = ''
    compartment = ''
    is_external = False
    direction = 'both'  # input, output, or both

    def __init__(self, id, name, compartment, is_external=False, direction='both'):
        self.id, self.name, self.compartment, self.is_external, self.direction = \
            id, name, compartment, is_external, direction


class Network:
    N = None
    reactions = []
    metabolites = []
    objective_reaction = None
    right_nullspace = None

    def __init__(self):
        self.reactions = []
        self.metabolites = []

    def reversible_reaction_indices(self):
        return [index for index, reaction in enumerate(self.reactions) if reaction.reversible]

    def external_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.is_external]

    def set_inputs(self, input_indices):
        for index in input_indices:
            self.metabolites[index].direction = 'input'

    def set_outputs(self, output_indices):
        for index in output_indices:
            self.metabolites[index].direction = 'output'

    def input_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.direction == 'input']

    def output_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.direction == 'output']

    def compress(self, verbose=False):
        if verbose:
            print('Compressing network')

        metabolite_count, reaction_count = self.N.shape

        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        self.cancel_singly(verbose=verbose)
        if verbose:
            print('Removed %d reactions and %d metabolites' %
                  (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        self.cancel_compounds(verbose=verbose)
        if verbose:
            print('Removed %d reactions and %d metabolites' %
                  (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

        ## This does not seem to do anything on the tested metabolic networks
        # if not self.right_nullspace:
        #     if verbose:
        #         print('Calculating null space')
        #     self.right_nullspace = np.transpose(helpers.nullspace(np.transpose(self.N), symbolic=False))
        #
        # self.remove_infeasible_irreversible_reactions(verbose=verbose)


        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        self.cancel_clementine(verbose=verbose)
        if verbose:
            print('Removed %d reactions and %d metabolites' %
                  (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

        if verbose:
            print('Removed %d reactions and %d metabolites in total' %
                  (reaction_count - self.N.shape[1], metabolite_count - self.N.shape[0]))
        pass

    def cancel_singly(self, verbose=False):
        """
        Urbanczik A4 T4
        :param verbose:
        :return:
        """
        if verbose:
            print('Trying to cancel compounds by singly produced/consumed metabolites')

        internal_metabolite_indices = np.setdiff1d(range(len(self.metabolites)), self.external_metabolite_indices())
        total_internal_metabolites = len(internal_metabolite_indices)

        for iteration, index in enumerate(internal_metabolite_indices):
            if verbose:
                print('Cancelling compounds - %.2f%%' % (iteration / float(total_internal_metabolites) * 100))

            reaction_index = None
            producing_reactions = list(np.where(self.N[index, :] > 0)[0])
            consuming_reactions = list(np.where(self.N[index, :] < 0)[0])
            reactions_to_cancel = []

            for reaction_index in producing_reactions:
                if self.reactions[reaction_index].reversible and reaction_index not in consuming_reactions:
                    consuming_reactions.append(reaction_index)

            for reaction_index in consuming_reactions:
                if self.reactions[reaction_index].reversible and reaction_index not in producing_reactions:
                    producing_reactions.append(reaction_index)

            if len(producing_reactions) == 1:
                # This internal metabolite is produced by only 1 reaction
                reaction_index = producing_reactions[0]
                reactions_to_cancel = consuming_reactions
                if verbose:
                    print('Metabolite %s is only produced in reaction %s. It will be cancelled through addition' % (self.metabolites[index].id, self.reactions[reaction_index].id))
            elif len(consuming_reactions) == 1:
                # This internal metabolite is consumed by only 1 reaction
                reaction_index = consuming_reactions[0]
                reactions_to_cancel = producing_reactions
                if verbose:
                    print('Metabolite %s is only consumed in reaction %s. It will be cancelled through addition' % (self.metabolites[index].id, self.reactions[reaction_index].id))
            else:
                continue

            for other_reaction_index in np.setdiff1d(reactions_to_cancel, [reaction_index]):
                factor = self.N[index, other_reaction_index] / self.N[index, reaction_index]
                self.N[:, other_reaction_index] = np.subtract(self.N[:, other_reaction_index],
                                                              self.N[:, reaction_index] * factor)

                if not self.reactions[reaction_index].reversible and self.reactions[other_reaction_index].reversible:
                    # Reactions changed by irreversible reactions must become irreversible too
                    self.reactions[other_reaction_index].reversible = False

        removable_metabolites, removable_reactions = [], []
        for metabolite_index in internal_metabolite_indices:
            if np.count_nonzero(self.N[metabolite_index, :]) == 1:
                # This metabolite is used in only one reaction
                reaction_index = [index for index in range(len(self.reactions)) if self.N[metabolite_index, index] != 0][0]
                removable_metabolites.append(metabolite_index)
                removable_reactions.append(reaction_index)

        self.drop_reactions(removable_reactions)
        self.drop_metabolites(removable_metabolites)

    def cancel_compounds(self, verbose=False):
        """
        Urbanczik A4 T3
        :param verbose:
        :return:
        """
        if verbose:
            print('Trying to cancel compounds by reversible reactions')

        internal_metabolite_indices = np.setdiff1d(range(len(self.metabolites)), self.external_metabolite_indices())
        reversible_reactions = self.reversible_reaction_indices()
        total_reversible_reactions = len(reversible_reactions)

        for iteration, reaction_index in enumerate(reversible_reactions):
            if verbose:
                print('Cancelling compounds - %.2f%%' % (iteration / float(total_reversible_reactions) * 100))
            reaction = self.N[:, reaction_index]
            metabolite_indices = [index for index in range(len(self.metabolites)) if reaction[index] != 0 and
                                  index not in self.external_metabolite_indices()]
            involved_in_reactions = [np.count_nonzero(self.N[index, :]) for index in metabolite_indices]

            if len(involved_in_reactions) == 0:
                # This reaction doesn't use any internal metabolites
                continue

            busiest_metabolite = np.argmax(involved_in_reactions)  # Involved in most reactions
            if not isinstance(busiest_metabolite, int) and not isinstance(busiest_metabolite, np.int64):
                busiest_metabolite = busiest_metabolite[0]

            # Heuristic: we choose to cancel the metabolite that is used in the largest number of other reactions
            target = metabolite_indices[busiest_metabolite]

            for other_reaction_index in range(self.N.shape[1]):
                # Make all other reactions that consume or produce target metabolite zero for that metabolite
                if other_reaction_index != reaction_index and self.N[target, other_reaction_index] != 0:
                    self.N[:, other_reaction_index] = np.subtract(self.N[:, other_reaction_index],
                                                                  self.N[:, reaction_index] * \
                                                                  (self.N[target, other_reaction_index] /
                                                                   self.N[target, reaction_index]))
                    # self.reactions[other_reaction_index].name = '(%s - %s)' % (self.reactions[other_reaction_index].name,
                    #                                                   reaction_index)

        removable_metabolites, removable_reactions = [], []
        for metabolite_index in internal_metabolite_indices:
            if np.count_nonzero(self.N[metabolite_index, :]) == 1:
                # This metabolite is used in only one reaction
                reaction_index = [index for index in range(len(self.reactions)) if self.N[metabolite_index, index] != 0][0]
                removable_metabolites.append(metabolite_index)
                removable_reactions.append(reaction_index)

        self.drop_reactions(removable_reactions)
        self.drop_metabolites(removable_metabolites)

    def cancel_clementine(self, verbose=False):
        if verbose:
            print('Compressing with SCEI')

        compressed_G = clementine_equality_compression(self.N, self.external_metabolite_indices(),
                                                       self.reversible_reaction_indices(), self.input_metabolite_indices(),
                                                       self.output_metabolite_indices())
        self.N = np.transpose(compressed_G)
        drop = []
        for row in range(self.N.shape[0]):
            if np.count_nonzero(self.N[row, :]) == 0:
                drop.append(row)
        self.drop_metabolites(drop)

    def remove_infeasible_irreversible_reactions(self, verbose=False):
        """
        Urbanczik A4 T1
        :param verbose:
        :return:
        """
        reversible = self.reversible_reaction_indices()
        irreversible_columns = [i for i in range(self.N.shape[1]) if i not in reversible]
        number_irreversible = len(irreversible_columns)
        reduced_nullspace = self.right_nullspace[:, irreversible_columns]

        if verbose:
            print('Removing infeasible irreversible reactions')

        c = [1] * number_irreversible
        result = linprog(c, A_eq=np.transpose(reduced_nullspace), b_eq=[0] * reduced_nullspace.shape[1],
                         bounds=([(0, 1)] * len(c)), options={'maxiter': 10000, 'disp': verbose},
                         method='simplex')

        if result.status > 0:
            if verbose:
                print('Linear programming optimisation failed with error: "%s"' % result.message)
            return

        removable_reactions = [irreversible_columns[index] for index, value in enumerate(result.x) if value > 0.001]
        if verbose:
            if len(removable_reactions):
                print('Removed the following infeasible irreversible reactions:')
                print('\t%s'.join([reaction.name for reaction in self.reactions[removable_reactions]]))
            else:
                print('No infeasible irreversible reactions found')

        if len(removable_reactions):
            self.drop_reactions(removable_reactions)

    def drop_reactions(self, reaction_indices):
        reactions_to_keep = [col for col in range(self.N.shape[1]) if col not in reaction_indices]
        self.N = self.N[:, reactions_to_keep]

        if len(self.reactions):
            self.reactions = [self.reactions[index] for index in reactions_to_keep]
        if self.right_nullspace is not None:
            # Since the right null space has as many rows as N has columns, we remove rows here
            self.right_nullspace = self.right_nullspace[reactions_to_keep, :]

    def drop_metabolites(self, metabolite_indices):
        metabolites_to_keep = [row for row in range(self.N.shape[0]) if row not in metabolite_indices]
        self.N = self.N[metabolites_to_keep, :]

        if len(self.metabolites):
            self.metabolites = [self.metabolites[index] for index in metabolites_to_keep]
