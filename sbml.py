from helpers import *
from time import time
from conversion_cone import get_conversion_cone

if __name__ == '__main__':
    start = time()

    # model_path = 'models/iAF1260.xml'
    # model_path = 'models/iND750.xml'
    # model_path = 'models/microbesflux_toy.xml'
    model_path = 'models/e_coli_core.xml'
    # model_path = 'models/e_coli_core_constr.xml'
    # model_path = 'models/e_coli_core_red.xml'
    # model_path = 'models/e_coli_core_nolac.xml'
    # model_path = 'models/daan_toy.xml'

    network = extract_sbml_stoichiometry(model_path)
    for index, item in enumerate(network.metabolites):
        print(index, item.id, item.name)

    symbolic = True
    inputs = [34, 54, 56, 60] # Glucose, ammonium, O2, phosphate
    # Acetate, acetaldehyde, 2-oxoglutarate, CO2, ethanol, formate, D-fructose, fumarate, L-glutamine, L-glutamate,
    # H2O, H+, lactate, L-malate, pyruvate, succinate
    # output_exceptions = [6, 8, 14, 19, 24, 28, 29, 31, 36, 38, 41, 43, 46, 48, 62, 69]
    output_exceptions = [24]
    outputs = np.setdiff1d(np.setdiff1d(network.external_metabolite_indices(), inputs), output_exceptions)
    c, H_cone, H = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                                       verbose=True, symbolic=symbolic)
                                       # input_metabolites=inputs, output_metabolites=outputs, verbose=True, symbolic=symbolic)
    np.savetxt('conversion_cone.csv', c, delimiter=',')

    for index, ecm in enumerate(c):
        if not ecm[-1]:
            continue
        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%d %s\t\t->\t%f' % (metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
