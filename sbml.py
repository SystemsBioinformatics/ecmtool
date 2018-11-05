from scipy.optimize import linprog

from helpers import *
from time import time
from conversion_cone import get_conversion_cone
from humanfriendly import format_timespan

if __name__ == '__main__':
    start = time()

    # model_path = 'models/iAF1260.xml'
    # model_path = 'models/iND750.xml'
    # model_path = 'models/microbesflux_toy.xml'
    # model_path = 'models/e_coli_core.xml'
    # model_path = 'models/e_coli_core_constr.xml'
    # model_path = 'models/e_coli_core_red.xml'
    # model_path = 'models/e_coli_core_nolac.xml'
    # model_path = 'models/daan_toy.xml'
    model_path = 'models/sxp_toy.xml'
    # model_path = 'models/sabp_compression.xml'

    network = extract_sbml_stoichiometry(model_path)

    for index, item in enumerate(network.metabolites):
        print(index, item.id, item.name)

    # inputs = [12, 21, 22, 24]  # Glucose, ammonium, O2, phosphate
    inputs = [34, 54, 56, 60]  # Glucose, ammonium, O2, phosphate
    # ignored_externals = [24, 6] # Ethanol and Acetate
    ignored_externals = list([])
    # ignored_externals = list(np.setdiff1d(network.external_metabolite_indices(), inputs + [6, 17, 18, 31]))  # CO2 H2O H+ biomass
    # ignored_externals = list(np.setdiff1d(network.external_metabolite_indices(), inputs + [19, 41, 43, 72]))  # CO2 H2O H+ biomass

    # for ignored in ignored_externals:
    #     id = 'ign_%d' % ignored
    #     network.reactions.append(Reaction(id, id, True))
    #     reaction = ([0] * ignored) + [1] + ([0] * (len(network.metabolites) - (ignored + 1)))
    #     network.N = np.append(network.N, np.transpose(np.asarray([reaction])), axis=1)
    #     network.metabolites[ignored].compartment = 'x'

    # Remove ignored metabolites
    network.N = np.delete(network.N, ignored_externals, axis=0)
    network.metabolites = np.delete(network.metabolites, ignored_externals, axis=0)

    orig_ids = [m.id for m in network.metabolites]
    orig_N = network.N
    # network.compress(verbose=True)
    # add_debug_tags(network)

    symbolic = True
    tagged_externals = network.external_metabolite_indices()

    c, H = get_conversion_cone(network.N, tagged_externals, network.reversible_reaction_indices(),
                               verbose=True, symbolic=symbolic)

    expanded_c = np.zeros(shape=(c.shape[0], len(orig_ids)))
    for column, id in enumerate([m.id for m in network.metabolites]):
        orig_column = [index for index, orig_id in enumerate(orig_ids) if orig_id == id][0]
        expanded_c[:, orig_column] = c[:, column]

    np.savetxt('conversion_cone.csv', expanded_c, delimiter=',')

    for index, ecm in enumerate(c):
        # if not ecm[-1]:
        #     continue
        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%d %s\t\t->\t%.4f' % (metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))
        # solution = linprog(c=[0] * orig_N.shape[1], A_eq=orig_N, b_eq=expanded_c[index,:], bounds=[(-1000, 1000)] * orig_N.shape[1])
        # print('ECM satisfies stoichiometry' if solution.status == 0 else 'ECM does not satisfy stoichiometry')

    end = time()
    print('Ran in %s' % format_timespan(end - start))
    pass
