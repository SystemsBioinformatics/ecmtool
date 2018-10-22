import numpy as np

def gen_random_network(amount_metabolites, ratio_external=0.1, ratio_external_input=0.1,  ratio_external_output=0.1):
    metabolites = {}
    amount_external = np.max([1, int(amount_metabolites * ratio_external)])
    amount_internal = np.max([1, amount_metabolites - amount_external])
    amount_input = np.max([1, int(amount_external * ratio_external_input)])
    amount_output = np.max([1, int(amount_external * ratio_external_output)])
    amount_in_output = np.max([1, amount_external - amount_input - amount_output])

    naming = {
        'IM_': amount_internal,
        'EM_IN_': amount_input,
        'EM_OUT_': amount_output,
        'EM_': amount_in_output,
    }

    for template, amount in naming.items():
        for index in range(1, amount+1):
            name = '%s%d' % (template, index)
            metabolites[name] = {
                'id': name,
                'boundary': 'EM_' in name,
                'SUBSYSTEM': 'c' if 'EM_' not in name else 'e'
            }
