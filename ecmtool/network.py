from fractions import Fraction

import cbmpy
import libsbml as sbml
import numpy as np
from scipy.linalg import null_space

from ecmtool.helpers import mp_print
from ecmtool.intersect_directly_mpi import setup_cycle_LP, perturb_LP, cycle_check_with_output, \
    independent_rows_qr, get_basis_columns_qr, remove_cycles, reaction_infeasibility_check
from .helpers import to_fractions, redund


def clementine_equality_compression(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[],
                                    output_metabolites=[],
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
    :param verbose:
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
            mp_print('\nIteration %d/%d' % (index, len(internal_metabolites)))

        # Find reactions that use this metabolite
        active_reactions = np.asarray([reaction_index for reaction_index in range(G.shape[0])
                                       if G[reaction_index, internal_metabolite] != 0])

        # Skip internal metabolites that aren't used anywhere
        if len(active_reactions) == 0:
            if verbose:
                mp_print(
                    'Skipping internal metabolite #%d, since it is not used by any reaction\n' % internal_metabolite)
            continue

        # Skip internal metabolites that are used too often (>= busy_threshold)
        busy_threshold = 10
        if len(active_reactions) >= busy_threshold:
            if verbose:
                mp_print(
                    'Skipping internal metabolite #%d, since it is used by too many reactions\n' % internal_metabolite)
            continue

        # Project conversions that use this metabolite onto the hyperplane internal_metabolite = 0
        projections = np.dot(G[active_reactions, :], equalities[index])
        positive = active_reactions[np.argwhere(projections > 0)[:, 0]]
        negative = active_reactions[np.argwhere(projections < 0)[:, 0]]
        candidates = np.ndarray(shape=(0, number_metabolites))

        if verbose:
            mp_print('Adding %d candidates' % (len(positive) * len(negative)))

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
            mp_print('Removing %d rays\n' % (G.shape[0] - len(keep)))
        G = G[keep, :]
        G = np.append(G, candidates, axis=0)
        # G = drop_nonextreme(G, get_zero_set(G, equalities), verbose=verbose)

        if candidates_since_last_redund > 100:
            if verbose:
                mp_print('Running redund')
            G = redund(G, verbose=verbose)
            candidates_since_last_redund = 0

    if candidates_since_last_redund > 0:
        if verbose:
            mp_print('Running redund')
        G = redund(G, verbose=verbose)

    return G


def get_sbml_model(path):
    doc = sbml.readSBMLFromFile(path)
    model = doc.getModel()
    model.__keep_doc_alive__ = doc  # Needed to prevent garbage collection from thrashing model
    return model


def extract_sbml_stoichiometry(path, add_objective=True, skip_external_reactions=True, determine_inputs_outputs=False,
                               use_external_compartment=None):
    """
    Parses an SBML file containing a metabolic network, and returns a Network instance
    with the metabolites, reactions, and stoichiometry initialised. By default will look
    for an SBML v3 FBC objective function, and skip reactions that contain '_EX_' in their ID.
    :param path: string absolute or relative path to the .sbml file
    :param add_objective: Look for SBML v3 FBC objective definition
    :param skip_external_reactions: Ignore external reactions, as identified by '_EX_' in their ID
    :param determine_inputs_outputs: Automatically determine input and output metabolites from analysing the network.
    :param use_external_compartment: If true, then all metabolites in the external compartment are marked as 'external',
            which means that they do not have to satisfy the steady-state constraint, and can pop up in the computed
            conversions. By default, ecmtool does not use this but does print a warning.
    :return: Network
    """
    cbmpy_model = cbmpy.readSBML3FBC(path)

    species = list(cbmpy_model.species)
    species_index = {item.id: index for index, item in enumerate(species)}
    reactions = cbmpy_model.reactions

    # Check if all reactions can run
    for reaction in reactions:
        reaction_feasible = True
        lowerBound, upperBound, _ = cbmpy_model.getFluxBoundsByReactionID(reaction.id)
        if lowerBound is not None and upperBound is not None:
            if lowerBound.value == 0 and upperBound.value == 0:
                reaction_feasible = False
            elif lowerBound.value > upperBound.value:
                reaction_feasible = False

        if not reaction_feasible:
            raise Exception(
                'Reaction {} has lower bound {} and upper bound {}, '
                'and is therefore infeasible. Please adjust bounds or delete reaction.'.format(
                    reaction.id, lowerBound.value, upperBound.value))

    objective_reaction_column = None
    pairs = cbmpy.CBTools.findDeadEndReactions(cbmpy_model)
    external_metabolites, external_reactions = list(zip(*pairs)) if len(pairs) else (
        list(zip(*cbmpy.CBTools.findDeadEndMetabolites(cbmpy_model)))[0], [])

    # Catch any metabolites that were not recognised automatically, but might be external
    for item in species:
        if use_external_compartment is None:
            if (item.compartment == 'e') & (item.id not in external_metabolites):
                mp_print('!!\n'
                         'Warning: metabolite %s appears to be in the external compartment, '
                         'but does not have an exchange reaction in the SBML-model. \n'
                         'We will impose the steady-state constraint on this metabolite, and it will '
                         'thus not appear in the computed conversion modes.\n'
                         'This behaviour can be changed with the ecmtool-argument \"use_external_compartment\"'
                         % item.id)

    if use_external_compartment is not None:
        external_metabolites_extended = list(external_metabolites) + [item.id for item in species if
                                                                      (item.compartment == use_external_compartment) & (
                                                                                  item.id not in external_metabolites)]
        external_metabolites = external_metabolites_extended

    network = Network()
    network.metabolites = [Metabolite(item.id, item.name, item.compartment, item.id in external_metabolites) for item
                           in species]
    # The following alternative is choosing only the metabolites ending with '_e' as external. This does not always work
    # because sometimes there are metabolites in the cytosol that still have a source or sink reaction
    # network.metabolites = [Metabolite(item.id, item.name, item.compartment, item.compartment == external_compartment)
    #                        for item in species]

    if add_objective:
        objective_name = cbmpy_model.getActiveObjective().fluxObjectives[0].reaction
        network.objective_reaction_id = objective_name

    if skip_external_reactions:
        reactions = [reaction for reaction in reactions if reaction.id not in external_reactions]

    if determine_inputs_outputs:
        for metabolite in [network.metabolites[index] for index in network.external_metabolite_indices()]:
            index = external_metabolites.index(metabolite.id)
            if use_external_compartment is not None:
                if index >= len(external_reactions):
                    mp_print(
                        'Warning: missing exchange reaction for metabolite %s. '
                        'Skipping marking this metabolite as input or output.\n '
                        'This behaviour can be changed with the ecmtool-argument \"use_external_compartment\"'
                        % metabolite.id)
                    continue

            reaction_id = external_reactions[index]
            reaction = cbmpy_model.getReaction(reaction_id)
            lowerBound, upperBound, _ = cbmpy_model.getFluxBoundsByReactionID(reaction_id)
            stoichiometries = reaction.getStoichiometry()
            stoichiometry = [stoich[0] for stoich in stoichiometries if stoich[1] == metabolite.id][0]

            if reaction.reversible:
                # Check if the reaction is truly bidirectional
                if (lowerBound is not None and lowerBound.value >= 0) or \
                        (upperBound is not None and upperBound.value <= 0):
                    reaction.reversible = False
                else:
                    # Reversible reactions are both inputs and outputs, so don't mark as either
                    continue

                if (lowerBound.value < 0 or lowerBound is None) and upperBound.value <= 0:
                    # Direction of model is inverted (substrates are products and vice versa. This happens sometimes,
                    # e.g. https://github.com/SBRG/bigg_models/issues/324
                    print(
                        'Swapping direction of reversible reaction %s that can only run in reverse direction.' % reaction_id)
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
        network.metabolites.append(
            Metabolite('objective', 'Virtual objective metabolite', 'e', is_external=True, direction='output'))
        N = np.append(N, to_fractions(np.zeros(shape=(1, N.shape[1]))), axis=0)
        N[-1, objective_reaction_column] = 1

    network.N = N
    network.objective_reaction_column = objective_reaction_column

    return network


def add_reaction_tags(network, reactions=[]):
    if len(reactions) == 0:
        reactions = range(len(network.reactions))

    tag_ids = []
    for reaction in reactions:
        tag_ids.append('virtual_tag_%s' % network.reactions[reaction].id)
        network.metabolites.append(Metabolite('virtual_tag_%s' % network.reactions[reaction].id,
                                              'Virtual tag for %s' % network.reactions[reaction].id,
                                              compartment='e', is_external=True,
                                              direction='both' if network.reactions[reaction].reversible else 'output'))
    network.N = np.append(network.N, to_fractions(np.identity(len(network.reactions)))[reactions, :], axis=0)

    return tag_ids


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

    def set_both(self, both_indices):
        for index in both_indices:
            self.metabolites[index].is_external = True
            self.metabolites[index].direction = 'both'

    def input_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if
                metabolite.direction == 'input' and metabolite.is_external]

    def output_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if
                metabolite.direction == 'output' and metabolite.is_external]

    def compress(self, verbose=False, SCEI=True, cycle_removal=False):
        """

        :param verbose:
        :param SCEI:
        :param cycle_removal: Cycle removal should only be done when input/output metabs are already split
        :return:
        """
        if verbose:
            mp_print('\n Compressing network')

        self.compressed = True
        self.uncompressed_metabolite_ids = [met.id for met in self.metabolites]
        self.uncompressed_metabolite_names = [met.name for met in self.metabolites]

        original_metabolite_count, original_reaction_count = self.N.shape
        original_internal = len(self.metabolites) - len(self.external_metabolite_indices())
        original_reversible = len(self.reversible_reaction_indices())
        no_fixed_point = True

        while no_fixed_point:
            before_met_count, before_reac_count = self.N.shape
            self.compress_inner(verbose=verbose, SCEI=SCEI, cycle_removal=cycle_removal)
            after_met_count, after_reac_count = self.N.shape
            if (after_met_count - before_met_count == 0) & (after_reac_count - before_reac_count == 0):
                no_fixed_point = False
            elif verbose:
                mp_print('\n Trying another compression cycle')

        if verbose:
            mp_print('\n Removed %d reactions and %d metabolites in total' %
                     (original_reaction_count - self.N.shape[1], original_metabolite_count - self.N.shape[0]))

            internal = len(self.metabolites) - len(self.external_metabolite_indices())
            reversible = len(self.reversible_reaction_indices())
            metabolite_count, reaction_count = self.N.shape
            mp_print('Original: %d metabolites (%d internal), %d reactions (%d reversible)' %
                     (original_metabolite_count, original_internal, original_reaction_count, original_reversible))
            mp_print('Compressed: %d metabolites (%d internal), %d reactions (%d reversible)' %
                     (metabolite_count, internal, reaction_count, reversible))
            mp_print('Compressed size: %.2f%%' % (((float(reaction_count) * metabolite_count) / (
                    original_reaction_count * original_metabolite_count)) * 100))

    def compress_inner(self, verbose=False, SCEI=True, cycle_removal=False):
        """

        :param verbose:
        :param SCEI:
        :param cycle_removal: Cycle removal should only be done when input/output metabolites are already split
        :return:
        """
        self.remove_infeasible_irreversible_reactions(verbose=verbose)
        self.cancel_compounds(verbose=verbose)
        self.cancel_singly(verbose=verbose)
        # self.cancel_dead_ends(verbose=verbose)
        if SCEI:
            self.cancel_clementine(verbose=verbose)
        self.remove_unused_metabs(verbose=verbose)
        self.split_reversible()
        self.N = np.transpose(redund(np.transpose(self.N)))
        self.reactions = [Reaction('R_%d' % i, 'Reaction %d' % i, reversible=False) for i in range(self.N.shape[1])]
        if cycle_removal:
            self.compress_cycles(verbose=verbose)
        self.reactions = [Reaction('R_%d' % i, 'Reaction %d' % i, reversible=False) for i in range(self.N.shape[1])]

    def compress_cycles(self, verbose=False):
        R, network, external_cycles = remove_cycles(self.N, self)
        n_reac_according_to_N = self.N.shape[1]
        removable_reacs = np.arange(n_reac_according_to_N, len(self.reactions))
        if len(removable_reacs):
            mp_print("Warning: in cycle removal, a reaction was removed without keeping track")
        return external_cycles

    def remove_unused_metabs(self, verbose=False):
        # Remove unused metabolites
        mp_print('Removing metabolites that are not used in any reaction.')
        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        removed = 0
        for i in np.flip(range(self.N.shape[0]), 0):
            if sum(abs(self.N[i])) == 0:
                if not self.metabolites[i].is_external:
                    self.drop_metabolites([i], force_external=True)
                    removed += 1
        if verbose:
            mp_print('Removed %d reactions and %d metabolites' %
                     (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

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
        expanded = np.repeat(np.repeat(to_fractions(np.zeros(shape=(1, 1))), matrix.shape[0], axis=0),
                             len(self.uncompressed_metabolite_ids), axis=1)

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
        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        removable_metabolites, removable_reactions = [], []

        if verbose:
            mp_print('Trying to cancel compounds by singly produced/consumed metabolites')

        internal_metabolite_indices = np.setdiff1d(range(len(self.metabolites)), self.external_metabolite_indices())
        total_internal_metabolites = len(internal_metabolite_indices)

        for iteration, index in enumerate(internal_metabolite_indices):
            if verbose:
                mp_print('Cancelling compounds - %.2f%%' % (iteration / float(total_internal_metabolites) * 100))

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
                    mp_print('Metabolite %s is only produced in reaction %s. It will be cancelled through addition' % (
                        self.metabolites[index].id, self.reactions[reaction_index].id))
            elif len(consuming_reactions) == 1:
                # This internal metabolite is consumed by only 1 reaction
                reaction_index = consuming_reactions[0]
                reactions_to_cancel = producing_reactions
                if verbose:
                    mp_print('Metabolite %s is only consumed in reaction %s. It will be cancelled through addition' % (
                        self.metabolites[index].id, self.reactions[reaction_index].id))
            else:
                continue

            for other_reaction_index in np.setdiff1d(reactions_to_cancel, [reaction_index]):
                factor = self.N[index, other_reaction_index] / self.N[index, reaction_index]
                self.N[:, other_reaction_index] = np.subtract(self.N[:, other_reaction_index],
                                                              self.N[:, reaction_index] * factor)

                if not self.reactions[reaction_index].reversible and self.reactions[other_reaction_index].reversible:
                    # Reactions changed by irreversible reactions must become irreversible too
                    self.reactions[other_reaction_index].reversible = False

        for metabolite_index in internal_metabolite_indices:
            if np.count_nonzero(self.N[metabolite_index, :]) == 1:
                # This metabolite is used in only one reaction
                reaction_index = [index for index in range(self.N.shape[1]) if self.N[metabolite_index, index] != 0][0]
                removable_metabolites.append(metabolite_index)
                removable_reactions.append(reaction_index)

        self.drop_reactions(removable_reactions)
        self.drop_metabolites(removable_metabolites)
        if verbose:
            mp_print('Removed %d reactions and %d metabolites' %
                     (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

    def cancel_dead_ends(self, verbose=False):
        """
        Cancel metabolites that are either only produced or only consumed
        :param verbose:
        :return:
        """
        removable_metabolites = []

        if verbose:
            mp_print('Removes compounds that are only produced/consumed.')

        internal_metabolite_indices = np.setdiff1d(range(len(self.metabolites)), self.external_metabolite_indices())
        total_internal_metabolites = len(internal_metabolite_indices)

        for iteration, index in enumerate(internal_metabolite_indices):
            if verbose:
                mp_print('Cancelling dead ends - %.2f%%' % (iteration / float(total_internal_metabolites) * 100))

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

            if len(producing_reactions) == 0 or len(consuming_reactions) == 0:
                removable_metabolites.append(index)

        self.drop_metabolites(removable_metabolites)

    def cancel_compounds(self, verbose=False):
        """
        Urbanczik A4 T3
        :param verbose:
        :return:
        """
        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        removed_count = 0
        if verbose:
            mp_print('Trying to cancel compounds by reversible reactions')

        internal_metabolite_indices = np.setdiff1d(range(len(self.metabolites)), self.external_metabolite_indices())
        reversible_reactions = np.array(self.reversible_reaction_indices())
        total_reversible_reactions = len(reversible_reactions)

        for iteration, orig_reaction_index in enumerate(reversible_reactions):
            reaction_index = orig_reaction_index - removed_count
            if verbose:
                mp_print('Cancelling compounds - %.2f%%' % (iteration / float(total_reversible_reactions) * 100))
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

            if np.count_nonzero(self.N[target, :]) == 1 and self.N[target, reaction_index] != 0:
                # This removes the reversible reaction that was used to cancel the metabolite.
                # This is the only one left that produces the metabolite, so cannot run in steady-state
                self.drop_reactions([reaction_index])
                # Since the reaction is removed, all next reaction indices should be decreased by one
                removed_count = removed_count + 1
            else:
                mp_print("Something went wrong in cancelling by using reversible reactions.")

        removable_metabolites, removable_reactions = [], []

        for metabolite_index in internal_metabolite_indices:
            nonzero_count = np.count_nonzero(self.N[metabolite_index, :])
            if nonzero_count == 1:
                # This metabolite is used in only one reaction
                reaction_index = \
                    [index for index in range(len(self.reactions)) if self.N[metabolite_index, index] != 0][0]
                removable_reactions.append(reaction_index)
                pass

        self.drop_reactions(removable_reactions)

        for metabolite_index in internal_metabolite_indices:
            nonzero_count = np.count_nonzero(self.N[metabolite_index, :])
            if nonzero_count == 0:
                removable_metabolites.append(metabolite_index)

        self.drop_metabolites(removable_metabolites)

        if verbose:
            mp_print('Removed %d reactions and %d metabolites' %
                     (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

    def cancel_clementine(self, verbose=False):
        metabolite_count_intermediate, reaction_count_intermediate = self.N.shape
        if verbose:
            mp_print('Compressing with SCEI')

        compressed_G = clementine_equality_compression(self.N, self.external_metabolite_indices(),
                                                       self.reversible_reaction_indices(),
                                                       self.input_metabolite_indices(),
                                                       self.output_metabolite_indices())
        self.N = np.transpose(compressed_G)
        self.reactions = [Reaction('R_%d' % i, 'Reaction %d' % i, reversible=False) for i in range(self.N.shape[1])]
        drop = []
        for metabolite_index in range(self.N.shape[0]):
            if np.count_nonzero(self.N[metabolite_index, :]) == 0:
                drop.append(metabolite_index)
        self.drop_metabolites(drop)
        if verbose:
            mp_print('Removed %d reactions and %d metabolites' %
                     (reaction_count_intermediate - self.N.shape[1], metabolite_count_intermediate - self.N.shape[0]))

    def remove_infeasible_irreversible_reactions(self, verbose=False):
        """
        Urbanczik A4 T1
        :param verbose:
        :return:
        """
        n_reactions_before = len(self.reactions)
        if verbose:
            mp_print('Removing infeasible irreversible reactions')
        # Find rows that should be zero in stoichiometry
        internal_indices = [ind for ind, metab in enumerate(self.metabolites) if not metab.is_external]
        removable_present = True
        while removable_present:
            removable_reactions = []
            N_int = self.N[internal_indices, :]
            if np.min(N_int.shape) == 0:
                break
            else:
                nullspace_int = null_space(N_int.astype(dtype='double'))
            reacs_notin_nullspace = np.where(np.max(np.abs(nullspace_int), axis=1) < 1e-8)[0]
            if len(reacs_notin_nullspace):
                removable_reactions.extend(reacs_notin_nullspace)
                if verbose:
                    if len(removable_reactions):
                        mp_print('Removed ' + str(len(removable_reactions)) + ' reactions that were never used in '
                                                                              'steady state, with IDs:')
                        mp_print('\t'.join([self.reactions[ind].id for ind in removable_reactions]))
                self.drop_reactions(removable_reactions)
                removable_present = True
                continue

            reversible = self.reversible_reaction_indices()
            irreversible_columns = [i for i in range(self.N.shape[1]) if i not in reversible]
            reduced_nullspace = nullspace_int[irreversible_columns, :]

            indep_eq_matrix = independent_rows_qr((np.transpose(reduced_nullspace)))
            A_eq, b_eq, c, x0 = setup_cycle_LP(np.array(indep_eq_matrix, dtype='float'), only_eq=True)

            basis = get_basis_columns_qr(np.asarray(A_eq, dtype='float'), verbose=False)
            if len(basis) != A_eq.shape[0]:
                mp_print(
                    'No further infeasible irreversible reactions can be found, due to numerical issues. They will be removed later in the process.')
                removable_present = False
            else:
                b_eq, x0 = perturb_LP(b_eq, x0, A_eq, basis, 1e-8)
                removable_present, status, removable_indices = reaction_infeasibility_check(c,
                                                                                       np.asarray(A_eq, dtype='float'),
                                                                                       x0, basis,
                                                                                       find_all_entering=True,
                                                                                       verbose=False)
                if removable_present:
                    removable_reactions.extend(np.array(irreversible_columns)[removable_indices])
                if verbose:
                    if len(removable_reactions):
                        mp_print('Removed ' + str(
                            len(removable_reactions)) + ' infeasible irreversible reactions, with IDs:')
                        mp_print('\t'.join([self.reactions[ind].id for ind in removable_reactions]))
                    else:
                        mp_print('No further infeasible irreversible reactions found')

                if len(removable_reactions):
                    self.drop_reactions(removable_reactions)

        n_reactions_after = len(self.reactions)
        if verbose:
            mp_print("Removed a total of " + str(n_reactions_before - n_reactions_after) + " infeasible reactions.")

    def drop_reactions(self, reaction_indices):
        reactions_to_keep = [col for col in range(self.N.shape[1]) if col not in reaction_indices]
        self.N = self.N[:, reactions_to_keep]

        if len(self.reactions):
            self.reactions = [self.reactions[index] for index in reactions_to_keep]
        if self.right_nullspace is not None:
            # Since the right null space has as many rows as N has columns, we remove rows here
            self.right_nullspace = self.right_nullspace[reactions_to_keep, :]

    def drop_metabolites(self, metabolite_indices, force_external=False):

        # TODO: debug line
        do_not_drop = []
        if not force_external:
            for metab_index in metabolite_indices:
                if self.metabolites[metab_index].is_external:
                    do_not_drop.append(metab_index)
                    mp_print(
                        'Tried to remove external metabolite %s, which was denied. External metabolites must remain.' %
                        self.metabolites[metab_index].id)

        [metabolite_indices.remove(ind) for ind in do_not_drop]
        metabolites_to_keep = [row for row in range(self.N.shape[0]) if row not in metabolite_indices]
        self.N = self.N[metabolites_to_keep, :]

        if len(self.metabolites):
            self.metabolites = [self.metabolites[index] for index in metabolites_to_keep]

    def add_metabolite(self, id, name, compartment='c', is_external=False, direction='both'):
        self.metabolites.append(Metabolite(id, name, compartment, is_external, direction))
        new_row = [0] * self.N.shape[1]
        self.N = np.append(self.N, [new_row], axis=0)

    def get_or_create_hide_metabolites(self):
        """
        Returns the metabolite indices of the two virtual "empty" metabolites that
        hidden metabolites get sourced from and sink into. If none exist yet, this
        method creates them before returning.
        :return: Tuple of metabolite indices as (source, sink)
        """
        source, sink = None, None
        for i, metabolite in enumerate(self.metabolites):
            if metabolite.id == 'hidden_source':
                source = i
            elif metabolite.id == 'hidden_sink':
                sink = i

        if source is None:
            self.add_metabolite('hidden_source', 'Hidden metabolite source', 'e', True, 'input')
            source = self.N.shape[0] - 1
        if sink is None:
            self.add_metabolite('hidden_sink', 'Hidden metabolite sink', 'e', True, 'output')
            sink = self.N.shape[0] - 1

        return source, sink

    def hide(self, metabolite_indices):
        """
        Hides external metabolites by transforming them into internal metabolites through the
        addition of bidirectional exchange reactions.
        :param metabolite_indices: list of metabolite indices
        :return:
        """
        # This 'get_or_create hide_metabolites' can be used to keep track of metabolites that are hidden, but makes
        # the code a bit slower. Therefore, it is commented out.
        # source, sink = self.get_or_create_hide_metabolites()

        for index in metabolite_indices:
            self.metabolites[index].is_external = False

            column = to_fractions(np.zeros(shape=(self.N.shape[0], 1)))
            if self.metabolites[index].direction in ['output', 'both']:
                reaction_name = 'R_HIDDEN_EX_OUT_%s' % self.metabolites[index].id
                self.reactions.append(Reaction(reaction_name, reaction_name, reversible=False))
                column[index, 0] = -1
                # column[sink, 0] = 1
                self.N = np.append(self.N, column, axis=1)

            column = to_fractions(np.zeros(shape=(self.N.shape[0], 1)))
            if self.metabolites[index].direction in ['input', 'both']:
                reaction_name = 'R_HIDDEN_EX_IN_%s' % self.metabolites[index].id
                self.reactions.append(Reaction(reaction_name, reaction_name, reversible=False))
                column[index, 0] = 1
                # column[source, 0] = -1
                self.N = np.append(self.N, column, axis=1)

    def prohibit(self, metabolite_indices):
        """
        Prohibits input or output of external metabolites by marking them as internal metabolites.
        :param metabolite_indices: list of metabolite indices
        :return:
        """
        for index in metabolite_indices:
            self.metabolites[index].is_external = False

    def remove_objective_reaction(self):
        reaction_matches = [index for index, reaction in enumerate(self.reactions) if
                            reaction.id == self.objective_reaction_id]
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

    def split_reversible(self):
        for i in self.reversible_reaction_indices():
            self.N = np.append(self.N, np.transpose([-self.N[:, i]]), axis=1)
            self.reactions[i].reversible = False
            self.reactions.append(Reaction(self.reactions[i].id + "_rev", self.reactions[i].name + "_rev", False))

    def in_out(self, ind, metabolite):
        # determine if metabolite is both an input and output to the network
        input = False
        output = False
        for i in range(len(self.reactions)):
            if self.N[ind][i] < 0:
                input = True
            if self.N[ind][i] > 0:
                output = True

        return input and output

    def split_in_out(self, only_rays=False):
        # Add virtual input and output metabolites for external metabolites that are both input and output
        # Add virtual input metabolites for input metabolites, to enforce that they can't be used as output
        # Similar for output metabolites
        # The latter two should still be done if only_rays = True
        count = 0
        start_metabolites = len(self.metabolites)
        for i, metabolite in enumerate(self.metabolites):
            if i > start_metabolites - 1:  # dont re-split _in and _out virtual metabolites
                break
            if metabolite.is_external:
                make_internal = False  # boolean indicating if we have added virtual metabolite, and if we should thus make the original metab internal
                if not only_rays:
                    if metabolite.direction == 'both' or metabolite.direction == 'input':
                        make_internal = True
                        new_in = Metabolite(metabolite.id + "_virtin", metabolite.name + "_virtin",
                                            metabolite.compartment,
                                            is_external=True, direction='input')
                        self.metabolites.append(new_in)
                        exchange_in = Reaction(metabolite.id + " exch_in", metabolite.id + " exch_in", reversible=False)
                        self.reactions.append(exchange_in)
                        new_row = [[int(x)] for x in np.zeros(self.N.shape[1])]
                        self.N = np.append(self.N, np.transpose(np.asarray(new_row)), axis=0)
                        new_column = [[int(x)] for x in np.zeros(self.N.shape[0])]
                        new_column[i] = [int(1)]
                        new_column[-1] = [int(-1)]
                        self.N = np.append(self.N, np.asarray(new_column), axis=1)

                    if metabolite.direction == 'both' or metabolite.direction == 'output':
                        make_internal = True
                        new_out = Metabolite(metabolite.id + "_virtout", metabolite.name + "_virtout",
                                             metabolite.compartment,
                                             is_external=True, direction='output')
                        self.metabolites.append(new_out)
                        exchange_out = Reaction(metabolite.id + " exch_out", metabolite.id + " exch_out",
                                                reversible=False)
                        self.reactions.append(exchange_out)
                        new_row = [[int(x)] for x in np.zeros(self.N.shape[1])]
                        self.N = np.append(self.N, np.transpose(np.asarray(new_row)), axis=0)
                        new_column = [[int(x)] for x in np.zeros(self.N.shape[0])]
                        new_column[i] = [int(-1)]
                        new_column[-1] = [int(1)]
                        self.N = np.append(self.N, np.asarray(new_column), axis=1)
                else:  # if only_rays
                    if metabolite.direction == 'input':
                        make_internal = True
                        new_in = Metabolite(metabolite.id + "_virtin", metabolite.name + "_virtin",
                                            metabolite.compartment,
                                            is_external=True, direction='input')
                        self.metabolites.append(new_in)
                        exchange_in = Reaction(metabolite.id + " exch_in", metabolite.id + " exch_in", reversible=False)
                        self.reactions.append(exchange_in)
                        new_row = [[int(x)] for x in np.zeros(self.N.shape[1])]
                        self.N = np.append(self.N, np.transpose(np.asarray(new_row)), axis=0)
                        new_column = [[int(x)] for x in np.zeros(self.N.shape[0])]
                        new_column[i] = [int(1)]
                        new_column[-1] = [int(-1)]
                        self.N = np.append(self.N, np.asarray(new_column), axis=1)

                    if metabolite.direction == 'output':
                        make_internal = True
                        new_out = Metabolite(metabolite.id + "_virtout", metabolite.name + "_virtout",
                                             metabolite.compartment,
                                             is_external=True, direction='output')
                        self.metabolites.append(new_out)
                        exchange_out = Reaction(metabolite.id + " exch_out", metabolite.id + " exch_out",
                                                reversible=False)
                        self.reactions.append(exchange_out)
                        new_row = [[int(x)] for x in np.zeros(self.N.shape[1])]
                        self.N = np.append(self.N, np.transpose(np.asarray(new_row)), axis=0)
                        new_column = [[int(x)] for x in np.zeros(self.N.shape[0])]
                        new_column[i] = [int(-1)]
                        new_column[-1] = [int(1)]
                        self.N = np.append(self.N, np.asarray(new_column), axis=1)

                if make_internal:
                    metabolite.id = metabolite.id + '_int'
                    metabolite.is_external = False
                    metabolite.compartment = 'virtual'
                    count += 1

        self.N = to_fractions(self.N)
        mp_print("Split %d input/output external metabolites" % count)
