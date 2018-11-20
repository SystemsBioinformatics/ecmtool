from scipy.optimize import linprog

from helpers import *


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

    def input_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.direction == 'input']

    def output_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.direction == 'output']

    def compress(self, verbose=False):
        if verbose:
            print('Compressing network')

        metabolite_count, reaction_count = self.N.shape

        self.cancel_compounds(verbose=verbose)

        ## This does not seem to do anything on the tested metabolic networks
        # if not self.right_nullspace:
        #     if verbose:
        #         print('Calculating null space')
        #     self.right_nullspace = np.transpose(helpers.nullspace(np.transpose(self.N), symbolic=False))
        #
        # self.remove_infeasible_irreversible_reactions(verbose=verbose)

        if verbose:
            print('Removed %d reactions and %d metabolites' %
                  (reaction_count - self.N.shape[1], metabolite_count - self.N.shape[0]))
        pass

    def cancel_compounds(self, verbose=False):
        if verbose:
            print('Trying to cancel compounds by reversible reactions')

        reversible_reactions = self.reversible_reaction_indices()
        total_reversible_reactions = len(reversible_reactions)

        for iteration, reaction_index in enumerate(reversible_reactions):
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

            # We want to cancel the metabolite that is used in the largest amount of other reactions
            target = metabolite_indices[busiest_metabolite]

            for other_reaction_index in range(self.N.shape[1]):
                # Make all other reactions that consume or produce target metabolite zero for that metabolite
                if other_reaction_index != reaction_index and self.N[target, other_reaction_index] != 0:
                    self.N[:, other_reaction_index] = np.subtract(self.N[:, other_reaction_index],
                                                                  self.N[:, reaction_index] * \
                                                                  (self.N[target, other_reaction_index] /
                                                                   self.N[target, reaction_index]))
                    self.reactions[other_reaction_index].name = '(%s - %s)' % (self.reactions[other_reaction_index].name,
                                                                      reaction_index)

        removable_metabolites, removable_reactions = [], []
        for metabolite_index in range(len(self.metabolites)):
            if metabolite_index not in self.external_metabolite_indices() and \
                            np.count_nonzero(self.N[metabolite_index, :]) == 1:
                # This metabolite is used in only one reaction
                reaction_index = [index for index in range(len(self.reactions)) if self.N[metabolite_index, index] != 0][0]
                removable_metabolites.append(metabolite_index)
                removable_reactions.append(reaction_index)

        self.drop_reactions(removable_reactions)
        self.drop_metabolites(removable_metabolites)

    def remove_infeasible_irreversible_reactions(self, verbose=False):
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
