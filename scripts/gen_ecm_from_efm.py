import numpy as np
from ecmtool.helpers import extract_sbml_stoichiometry, to_fractions

if __name__ == '__main__':
    model_path = 'models/e_coli_core.xml'
    network_all = extract_sbml_stoichiometry(model_path, skip_external_reactions=False)
    internal_reaction_indices = [index for index, reaction in enumerate(network_all.reactions) if '_EX' not in reaction.id]

    efms = to_fractions(np.genfromtxt('e_coli_core_efms.csv', delimiter=','))

    ecms = np.dot(network_all.N[:, internal_reaction_indices], efms[internal_reaction_indices, :])
    ecms_unique = list({tuple(ecms[:, col]): 1 for col in range(ecms.shape[1])}.keys())
    pass