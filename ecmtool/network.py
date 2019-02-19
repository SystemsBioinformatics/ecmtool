from cdd import Fraction
import libsbml as sbml
import cbmpy

from scipy.optimize import linprog
import numpy as np

from .helpers import to_fractions, redund


def clementine_equality_compression(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                                    verbose=True):
    """
    Compresses a metabolic network using Superior Clementine Equality Intersection (all rights reserved).
    Follows the general Double Description method by Motzkin, using G as initial basis and intersecting
    hyperplanes of internal metabolites = 0.
    :param N:
    :param external_metabolites:
    :param reversible_reactions:
    :param input_metabolites:
    :param output_metabolites:
    :return:
    """
    number_metabolites, amount_reactions = N.shape[0], N.shape[1]
    internal_metabolites = np.setdiff1d(range(number_metabolites), external_metabolites)
    candidates_since_last_redund = 0

    identity = to_fractions(np.identity(number_metabolites))
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
            print('\nIteration %d/%d' % (index, len(internal_metabolites)))

        # Find reactions that use this metabolite
        active_reactions = np.asarray([reaction_index for reaction_index in range(G.shape[0])
                              if G[reaction_index, internal_metabolite] != 0])

        # Skip internal metabolites that aren't used anywhere
        if len(active_reactions) == 0:
            if verbose:
                print('Skipping internal metabolite #%d, since it is not used by any reaction\n' % internal_metabolite)
            continue

        # Skip internal metabolites that are used too often (>= busy_threshold)
        busy_threshold = 10
        if len(active_reactions) >= busy_threshold:
            if verbose:
                print('Skipping internal metabolite #%d, since it is used by too many reactions\n' % internal_metabolite)
            continue

        # Project conversions that use this metabolite onto the hyperplane internal_metabolite = 0
        projections = np.dot(G[active_reactions, :], equalities[index])
        positive = active_reactions[np.argwhere(projections > 0)[:, 0]]
        negative = active_reactions[np.argwhere(projections < 0)[:, 0]]
        candidates = np.ndarray(shape=(0, number_metabolites))

        if verbose:
            print('Adding %d candidates' % (len(positive) * len(negative)))

        # Make convex combinations of all pairs (positive, negative) such that their internal_metabolite = 0
        for pos in positive:
            for neg in negative:
                candidate = np.add(G[pos, :], G[neg, :] * (G[pos, internal_metabolite] / -G[neg, internal_metabolite]))
                if np.count_nonzero(candidate) > 0:
                    candidates = np.append(candidates, [candidate], axis=0)

        candidates_since_last_redund += len(candidates)

        # Keep only rays that satisfy internal_metabolite = 0
        keep = np.setdiff1d(range(G.shape[0]), active_reactions)
        if verbose:
            print('Removing %d rays\n' % (G.shape[0] - len(keep)))
        G = G[keep, :]
        G = np.append(G, candidates, axis=0)
        # G = drop_nonextreme(G, get_zero_set(G, equalities), verbose=verbose)

        if candidates_since_last_redund > 100:
            if verbose:
                print('Running redund')
            G = redund(G, verbose=verbose)
            candidates_since_last_redund = 0

    if candidates_since_last_redund > 0:
        if verbose:
            print('Running redund')
        G = redund(G, verbose=verbose)

    return G




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
    cbmpy_model = cbmpy.readSBML3FBC(path)

    species = list(cbmpy_model.species)
    species_index = {item.id: index for index, item in enumerate(species)}
    reactions = cbmpy_model.reactions
    objective_reaction_column = None
    pairs = cbmpy.CBTools.findDeadEndReactions(cbmpy_model)
    external_metabolites, external_reactions = zip(*pairs) if len(pairs) else (zip(*cbmpy.CBTools.findDeadEndMetabolites(cbmpy_model))[0], [])

    # Catch any metabolites that were not recognised automatically, but are likely external
    external_metabolites = list(external_metabolites) + [item.id for item in species if item.compartment == external_compartment]

    network = Network()
    network.metabolites = [Metabolite(item.id, item.name, item.compartment, item.id in external_metabolites) for item in species]

    if add_objective:
        objective_name = cbmpy_model.getActiveObjective().fluxObjectives[0].reaction
        network.objective_reaction_id = objective_name

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
                    reagents = cbmpy_model.getReaction(reaction_id).reagents
                    for met in reagents:
                        met.coefficient *= -1

            metabolite.direction = 'input' if stoichiometry >= 0 else 'output'

    # Build stoichiometry matrix N
    N = np.zeros(shape=(len(species), len(reactions)), dtype='object')
    for column, reaction in enumerate(reactions):
        network.reactions.append(Reaction(reaction.id, reaction.name, reaction.reversible))

        if add_objective and reaction.id == objective_name:
            objective_reaction_column = column

        reagents = reaction.reagents
        for metabolite in reagents:
            row = species_index[metabolite.species_ref]
            N[row, column] = Fraction(str(metabolite.coefficient))

    # Add objective metabolite from objective reaction
    if add_objective and objective_reaction_column is not None:
        network.metabolites.append(Metabolite('objective', 'Virtual objective metabolite', 'e', is_external=True, direction='output'))
        N = np.append(N, to_fractions(np.zeros(shape=(1, N.shape[1]))), axis=0)
        N[-1, objective_reaction_column] = 1

    network.N = N
    network.objective_reaction_column = objective_reaction_column

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
    right_nullspace = None
    compressed = False
    uncompressed_metabolite_ids = []
    uncompressed_metabolite_names = []
    objective_reaction_id = None
    objective_reaction_stoich = []

    def __init__(self):
        self.reactions = []
        self.metabolites = []

    def reversible_reaction_indices(self):
        return [index for index, reaction in enumerate(self.reactions) if reaction.reversible]

    def external_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.is_external]

    def set_inputs(self, input_indices):
        for index in input_indices:
            self.metabolites[index].is_external = True
            self.metabolites[index].direction = 'input'

    def set_outputs(self, output_indices):
        for index in output_indices:
            self.metabolites[index].is_external = True
            self.metabolites[index].direction = 'output'

    def input_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.direction == 'input' and metabolite.is_external]

    def output_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.direction == 'output' and metabolite.is_external]

    def compress(self, verbose=False):
        if verbose:
            print('Compressing network')

        self.compressed = True
        self.uncompressed_metabolite_ids = [met.id for met in self.metabolites]
        self.uncompressed_metabolite_names = [met.name for met in self.metabolites]

        original_metabolite_count, original_reaction_count = self.N.shape
        original_internal = len(self.metabolites) - len(self.external_metabolite_indices())
        original_reversible = len(self.reversible_reaction_indices())

        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        self.cancel_compounds(verbose=verbose)
        if verbose:
            print('Removed %d reactions and %d metabolites' %
                  (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        self.cancel_singly(verbose=verbose)
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
                  (original_reaction_count - self.N.shape[1], original_metabolite_count - self.N.shape[0]))

            internal = len(self.metabolites) - len(self.external_metabolite_indices())
            reversible = len(self.reversible_reaction_indices())
            metabolite_count, reaction_count = self.N.shape
            print('Original: %d metabolites (%d internal), %d reactions (%d reversible)' %
                  (original_metabolite_count, original_internal, original_reaction_count, original_reversible))
            print('Compressed: %d metabolites (%d internal), %d reactions (%d reversible)' %
                  (metabolite_count, internal, reaction_count, reversible))
            print('Compressed size: %.2f%%' % (((float(reaction_count) * metabolite_count) / (original_reaction_count * original_metabolite_count)) * 100))

        pass

    def uncompress(self, matrix):
        """
        Adds metabolite columns that were removed during compression to given
        matrix of metabolite variables.
        :param matrix: z by (m-n) matrix, with m number of original metabolites, and n removed during compression
        :return: z by m matrix
        """
        if not self.compressed:
            # No compression was performed
            return matrix

        # expanded = to_fractions(np.zeros(shape=(matrix.shape[0], len(self.uncompressed_metabolite_ids))))
        expanded = np.repeat(np.repeat(to_fractions(np.zeros(shape=(1, 1))), matrix.shape[0], axis=0), len(self.uncompressed_metabolite_ids), axis=1)

        for column, id in enumerate([m.id for m in self.metabolites]):
            orig_column = [index for index, orig_id in
                           enumerate(self.uncompressed_metabolite_ids) if orig_id == id][0]
            expanded[:, orig_column] = matrix[:, column]

        return expanded

    def append_uncompressed_stoich(self, matrix):
        """
        Appends uncompressed reactions to compressed network
        :param matrix:
        :return:
        """
        compressed_ids = [met.id for met in self.metabolites]
        mapping = [index for index, id in enumerate(self.uncompressed_metabolite_ids) if id in compressed_ids]
        self.N = np.append(self.N, matrix[mapping, :], axis=1)

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

            least_used_metabolite = np.argmin(involved_in_reactions)  # Involved in least reactions
            # argmin returns list if equal values are found. In this case, save the first one
            if not isinstance(least_used_metabolite, int) and not isinstance(least_used_metabolite, np.int64):
                least_used_metabolite = least_used_metabolite[0]

            # Heuristic: we choose to cancel the metabolite that is used in the smallest number of other reactions
            target = metabolite_indices[least_used_metabolite]

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
            nonzero_count = np.count_nonzero(self.N[metabolite_index, :])
            if nonzero_count == 1:
                # This metabolite is used in only one reaction
                reaction_index = [index for index in range(len(self.reactions)) if self.N[metabolite_index, index] != 0][0]
                removable_reactions.append(reaction_index)

        self.drop_reactions(removable_reactions)

        for metabolite_index in internal_metabolite_indices:
            nonzero_count = np.count_nonzero(self.N[metabolite_index, :])
            if nonzero_count == 0:
                removable_metabolites.append(metabolite_index)

        self.drop_metabolites(removable_metabolites)

    def cancel_clementine(self, verbose=False):
        if verbose:
            print('Compressing with SCEI')

        compressed_G = clementine_equality_compression(self.N, self.external_metabolite_indices(),
                                                       self.reversible_reaction_indices(), self.input_metabolite_indices(),
                                                       self.output_metabolite_indices())
        self.N = np.transpose(compressed_G)
        self.reactions = [Reaction('R_%d' % i, 'Reaction %d' % i, reversible=False) for i in range(self.N.shape[1])]
        drop = []
        for metabolite_index in range(self.N.shape[0]):
            if np.count_nonzero(self.N[metabolite_index, :]) == 0:
                drop.append(metabolite_index)
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

        #TODO: debug line
        for index in metabolite_indices:
            if self.metabolites[index].is_external:
                print('Tried to remove external metabolite %s, which was denied. External metabolites must remain.' % self.metabolites[index].id)
                return

        metabolites_to_keep = [row for row in range(self.N.shape[0]) if row not in metabolite_indices]
        self.N = self.N[metabolites_to_keep, :]

        if len(self.metabolites):
            self.metabolites = [self.metabolites[index] for index in metabolites_to_keep]

    def hide(self, metabolite_indices):
        """
        Hides external metabolites by transforming them into internal metabolites through the
        addition of bidirectional exchange reactions.
        :param metabolite_indices: list of metabolite indices
        :return:
        """
        for index in metabolite_indices:
            self.metabolites[index].is_external = False
            reaction_name = 'R_HIDDEN_EX_%s' % self.metabolites[index].id
            reversible = self.metabolites[index].direction == 'both'

            self.reactions.append(Reaction(reaction_name, reaction_name, reversible=reversible))
            row = to_fractions(np.zeros(shape=(self.N.shape[0], 1)))
            row[index, 0] += -1 if self.metabolites[index].direction == 'output' else 1
            self.N = np.append(self.N, row, axis=1)

    def remove_objective_reaction(self):
        reaction_matches = [index for index, reaction in enumerate(self.reactions) if reaction.id == self.objective_reaction_id]
        if len(reaction_matches) == 0:
            return

        reaction_index = reaction_matches[0]
        reaction = self.N[:, reaction_index]
        self.objective_reaction_stoich = reaction

        reactants = np.where(reaction < 0)[0]
        products = np.where(reaction > 0)[0]

        for reactant in reactants:
            self.metabolites[reactant].orig_direction = self.metabolites[reactant].direction
            self.metabolites[reactant].orig_external = self.metabolites[reactant].is_external
            self.metabolites[reactant].direction = 'output'
            self.metabolites[reactant].is_external = True

        for product in products:
            self.metabolites[product].orig_direction = self.metabolites[product].direction
            self.metabolites[product].orig_external = self.metabolites[product].is_external
            self.metabolites[product].direction = 'input'
            self.metabolites[product].is_external = True

        self.drop_reactions([reaction_index])

    def restore_objective_reaction(self):
        if self.objective_reaction_id is None:
            return

        stoich = self.objective_reaction_stoich
        if len(stoich) != len(self.metabolites):
            # Network is compressed, and objective reaction is not
            self.append_uncompressed_stoich(np.transpose([self.objective_reaction_stoich]))
            stoich = self.N[:, -1]
        else:
            self.N = np.append(self.N, np.transpose([self.objective_reaction_stoich]), axis=1)

        self.reactions.append(Reaction('objective', 'Objective', False))
        reactants = np.where(stoich < 0)[0]
        products = np.where(stoich > 0)[0]

        for reactant in reactants:
            self.metabolites[reactant].direction = self.metabolites[reactant].orig_direction
            self.metabolites[reactant].is_external = self.metabolites[reactant].orig_external

        for product in products:
            self.metabolites[product].direction = self.metabolites[product].orig_direction
            self.metabolites[product].is_external = self.metabolites[product].orig_external

    def get_objective_reagents(self):
        # TODO: add support for when biomass reaction isn't the last reaction
        # (i.e. when restore_objective_function() has not just been called.
        stoich = self.N[:, -1]
        reactants = list(np.where(stoich < 0)[0])
        products = list(np.where(stoich > 0)[0])
        return reactants, products