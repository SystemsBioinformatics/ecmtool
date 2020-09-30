import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from ecmtool import extract_sbml_stoichiometry, get_conversion_cone

from examples_and_results.helpers_ECM_calc import get_efms

"""
This script calculates the EFMs and ECMs for a number of models (created by taking subnetworks of e_coli_core).
Note that this script only works if EFMtool and Matlab is installed and working on your machine.
"""
EFMTOOL_PATH = 'C:\\Users\\Daan\\surfdrive\\PhD\\Software\\efmtool'

models = ['active_subnetwork_KO_' + str(i) for i in range(9)]
models = models + ['active_subnetwork', 'active_subnetwork_FVA', 'e_coli_core']
models=['e_coli_core']

number_info_df = pd.DataFrame(columns=['model', 'ECMs', 'EFMs', 'n_reacs'])

for model_str in models:
    model_path = os.path.join('models', model_str+'.xml')

    """Check if efms match ecms in E. coli core"""
    # Load model in ecmtool and create network object
    full_model = extract_sbml_stoichiometry(model_path, add_objective=True,
                                            determine_inputs_outputs=True,
                                            skip_external_reactions=True)

    network = full_model

    ex_N = full_model.N
    identity = np.identity(len(full_model.metabolites))
    reversibilities = [reaction.reversible for reaction in full_model.reactions]

    # Create appropriate stoichiometric matrix N, and reversibilities vector
    # Add exchange reactions so efmtool can calculate EFMs in steady state
    n_exchanges = 0
    reac_ids = [reac.id for reac in full_model.reactions]
    for index, metabolite in enumerate(full_model.metabolites):
        if metabolite.is_external:
            reaction = identity[:, index] if metabolite.direction != 'output' else -identity[index]
            ex_N = np.append(ex_N, np.transpose([reaction]), axis=1)
            reac_ids.append('R_ex_'+metabolite.id)
            reversibilities.append(True if metabolite.direction == 'both' else False)
            n_exchanges += 1

    efms = get_efms(ex_N, reversibilities, verbose=True,
                    efmtool_path=EFMTOOL_PATH)

    """"Split in and out metabolites, to facilitate ECM computation"""
    network.split_in_out(only_rays=False)

    network.compress(verbose=True)

    """Stap 3: Ecms enumereren"""
    cone = network.uncompress(
        get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                            network.input_metabolite_indices(), network.output_metabolite_indices(), only_rays=False,
                            verbose=True))

    cone = cone.transpose()

    n_efms = efms.shape[0]
    n_ecms = cone.shape[1]
    number_info_df = number_info_df.append(
        {'model': model_str, 'ECMs': n_ecms, 'EFMs': n_efms, 'n_reacs': ex_N.shape[1]}, ignore_index=True)

number_info_df_td = pd.melt(number_info_df, id_vars=['model','n_reacs'], var_name='type', value_name='number')
number_info_df_td.n_reacs = number_info_df_td.n_reacs.astype(float)
number_info_df_td.number = number_info_df_td.number.astype(float)

ax = sns.scatterplot(x='n_reacs', y='number', hue='type', data=number_info_df_td, s=160)
ax.set(xlabel='Number of reactions in selected subnetwork', ylabel='Number of modes')
ax.set(yscale='log')

plt.savefig(os.path.join(os.getcwd(), "result_files", "comparison_n_ecms_efms.png"))
plt.savefig(os.path.join(os.getcwd(), "result_files", "comparison_n_ecms_efms.svg"))
pass
