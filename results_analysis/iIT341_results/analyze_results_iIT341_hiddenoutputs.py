import pandas as pd
import csv
from results_analysis.helpers_analyze_results import *
import numpy as np
from ecmtool.network import extract_sbml_stoichiometry

model_path = 'iIT341.xml'

# inputs = [22, 33, 35, 93, 294, 300, 306, 314, 334, 356, 231, 262, 139, 28]
# hides = [16, 26, 29, 33, 39, 40, 59, 65, 75, 81, 90, 93, 100, 110, 145, 171, 174, 212, 223, 224, 232, 234, 235, 239,
#         252, 253, 255, 259, 261, 262, 263, 265, 269, 271, 276, 277, 279, 280, 283, 284, 286, 291, 293, 296, 302, 308,
#         312, 319, 320, 323, 325, 329, 331, 336, 341, 342, 344, 345, 350, 352, 358, 361, 366, 368, 370, 372]

"""Check if efms match ecms in E. coli core"""
# Load model in ecmtool and create network object
full_model = extract_sbml_stoichiometry(model_path, add_objective=True,
                                        determine_inputs_outputs=True,
                                        skip_external_reactions=True)

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
        reac_ids.append('R_ex_' + metabolite.id)
        reversibilities.append(True if metabolite.direction == 'both' else False)
        n_exchanges += 1

# Store N and reversibilities to solve it with Matlab
ids = [metab.id for metab in full_model.metabolites]
np.savetxt('ex_N_iIT341.csv', ex_N, delimiter=',')
np.savetxt('reversibilities_iIT341.csv', reversibilities, delimiter=',')

with open('metab_ids_iIT341.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(ids)

with open('reac_ids_iIT341.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(reac_ids)

# Import ECMs and EFMs
ecms_transposed_df = pd.read_csv('20200316_iIT341_hidden_outputs_conversion_cone.csv', header=0)
metab_ids = list(ecms_transposed_df)

ecms = normalize_columns_fractions(np.transpose(ecms_transposed_df.values))
ecms_df = pd.DataFrame(data=ecms, index=metab_ids)

np.savetxt('20200316_ecms_iIT341_hidden_outputs.csv', np.transpose(ecms), delimiter=',', header=','.join(metab_ids), comments='')

pass
