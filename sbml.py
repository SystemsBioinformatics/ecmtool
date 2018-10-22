import json

import matplotlib.pyplot as plt

import libsbml as sbml
from sympy import solve_linear_system, Matrix, symbols
from sympy.abc import x

from fba import do_fba
from helpers import *
from time import time
from conversion_cone import get_conversion_cone

if __name__ == '__main__':
    start = time()

    plt.bar

    # model_path = 'models/iAF1260.xml'
    # model_path = 'models/iND750.xml'
    # model_path = 'models/microbesflux_toy.xml'
    model_path = 'models/e_coli_core.xml'
    # model_path = 'models/e_coli_core_constr.xml'
    # model_path = 'models/e_coli_core_red.xml'
    # model_path = 'models/e_coli_core_nolac.xml'
    # model_path = 'models/daan_toy.xml'

    network = extract_sbml_stoichiometry(model_path, add_objective=False)
    # network_all = extract_sbml_stoichiometry(model_path, skip_external_reactions=False)
    for index, item in enumerate(network.metabolites):
        print(index, item.id, item.name)

    # add_debug_tags(network)
    # network.compress(verbose=True)

    inputs = [34, 54, 56, 60] # Glucose, ammonium, O2, phosphate
    c, H_cone, H = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                                       input_metabolites=inputs, output_metabolites=np.setdiff1d(network.external_metabolite_indices(), inputs), verbose=True)

    # TODO: REMOVE DEBUG BLOCK
    # fba = do_fba(model_path)
    # fba_fluxes = np.zeros(shape=(1, len(network_all.reactions)))
    # for key, value in fba.items():
    #     indices = [index for index, reaction in enumerate(network_all.reactions) if reaction.id == key]
    #
    #     if not len(indices):
    #         continue
    #
    #     reaction_index = indices[0]
    #     fba_fluxes[0, reaction_index] = value
    #
    # biomass_flux = fba['R_BIOMASS_Ecoli_core_w_GAM']
    # fba_normalised = {key: value/biomass_flux for key,value in fba.items()}

    # d_c = np.dot(network.N, np.transpose(to_fractions(fba_fluxes)))
    # H_part_c = np.dot(H_cone, d_c)
    # H_c = np.dot(H, d_c)

    np.savetxt('conversion_cone.csv', c, delimiter=',')

    for index, ecm in enumerate(c):
        if not ecm[-1]:
            continue
        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%d %s\t\t->\t%f' % (metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

        # result = solve_linear_system(Matrix(np.append(network.N, np.transpose(np.asmatrix(ecm, dtype='object')), axis=1)), *symbols(' '.join([str(x) for x in range(len(ecm))])))
        # result = np.linalg.solve(np.asmatrix(network.N, dtype='float64'), np.transpose(np.asmatrix(ecm, dtype='float64')))
        # correct = np.allclose(np.dot(network.N, result), ecm)
        # print('ECM is %s' % ('correct' if correct else 'incorrect'))

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
