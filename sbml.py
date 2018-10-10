import json

import libsbml as sbml
from helpers import *
from time import time
from conversion_cone import get_conversion_cone

if __name__ == '__main__':
    start = time()

    # model_path = 'models/iAF1260.xml'
    # model_path = 'models/iND750.xml'
    # model_path = 'models/microbesflux_toy.xml'
    model_path = 'models/e_coli_core.xml'

    network = extract_sbml_stoichiometry(model_path)
    network_all = extract_sbml_stoichiometry(model_path, skip_external_reactions=False)
    for index, item in enumerate(network.metabolites):
        print(index, item.id, item.name)

    # add_debug_tags(network)
    # network.compress(verbose=True)

    c, H_cone, H = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(), verbose=True)

    # TODO: REMOVE DEBUG BLOCK
    fba = json.loads('\n'.join(open('e_coli_core_fba.json').readlines()))
    fba_fluxes = np.zeros(shape=(1, len(network.reactions)))
    for key, value in fba.items():
        if 'EX_' in key:
            continue

        indices = [index for index, reaction in enumerate(network.reactions) if reaction.id == key]

        if not len(indices):
            continue

        reaction_index = indices[0]
        fba_fluxes[0, reaction_index] = value

    d_c = np.dot(network.N, np.transpose(to_fractions(fba_fluxes)))
    H_part_c = np.dot(H_cone, d_c)
    H_c = np.dot(H, d_c)

    np.savetxt('conversion_cone.csv', c, delimiter=',')

    for index, ecm in enumerate(c):
        if not ecm[72] or not ecm[34]:
            continue
        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%d %s\t\t->\t%f' % (metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
