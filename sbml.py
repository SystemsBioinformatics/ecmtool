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
    for index, item in enumerate(network.metabolites):
        print(index, item.id, item.name)

    # add_debug_tags(network)
    # network.compress(verbose=True)

    c = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(), verbose=True)

    np.savetxt('conversion_cone.csv', c, delimiter=',')

    for index, ecm in enumerate(c):
        # if not ecm[72] or not ecm[34]:
        #     continue
        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%d %s\t\t->\t%f' % (metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
