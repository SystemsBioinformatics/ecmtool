import numpy as np

from helpers import *
import helpers
from scipy.optimize import linprog


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

    def __init__(self, id, name, compartment):
        self.id, self.name, self.compartment = id, name, compartment


class Network:
    N = None
    reactions = []
    metabolites = []
    objective_reaction = None
    right_nullspace = None

    def reversible_reaction_indices(self):
        return [index for index, reaction in enumerate(self.reactions) if reaction.reversible]

    def external_metabolite_indices(self):
        return [index for index, metabolite in enumerate(self.metabolites) if metabolite.compartment == 'e']

    def compress(self, verbose=False):
        if verbose:
            print('Compressing network')
        if not self.right_nullspace:
            if verbose:
                print('Calculating null space')
            self.right_nullspace = helpers.nullspace(np.transpose(self.N))

        self.remove_infeasible_irreversible_reactions(verbose=verbose)
        pass

    def remove_infeasible_irreversible_reactions(self, verbose=False):
        reversible = self.reversible_reaction_indices()
        irreversible_rows = [i for i in range(self.N.shape[0]) if i not in reversible]
        number_irreversible = len(irreversible_rows)
        reduced_nullspace = self.right_nullspace[irreversible_rows, :]

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

        removable_reactions = [irreversible_rows[index] for index, value in enumerate(result.x) if value > 0.001]
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
            self.reactions = self.reactions[reactions_to_keep]
        if self.right_nullspace:
            # Since the right null space has as many rows as N has columns, we remove rows here
            self.right_nullspace = self.right_nullspace[reactions_to_keep, :]

    def drop_metabolites(self, metabolite_indices):
        metabolites_to_keep = [row for row in range(self.N.shape[0]) if row not in metabolite_indices]
        self.N = self.N[metabolites_to_keep, :]

        if len(self.metabolites):
            self.metabolites = self.metabolites[metabolites_to_keep]