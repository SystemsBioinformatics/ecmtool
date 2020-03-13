import pandas as pd
import csv
from helpers_analyze_results import *
import numpy as np
from ecmtool.network import extract_sbml_stoichiometry

model_path = 'models/iAB_RBC_283.xml'

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
        reac_ids.append('R_ex_'+metabolite.id)
        reversibilities.append(True if metabolite.direction == 'both' else False)
        n_exchanges += 1

# Store N and reversibilities to solve it with Matlab
ids = [metab.id for metab in full_model.metabolites]
np.savetxt('tmp/ex_N.csv', ex_N, delimiter=',')
np.savetxt('tmp/reversibilities.csv',reversibilities, delimiter=',')

with open('tmp/metab_ids.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(ids)

with open('tmp/reac_ids.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(reac_ids)


# Import ECMs and EFMs

# Use check_bijection_csvs()


# old_inout_ecms = pd.read_csv('KO8_old_inout_conversion_cone.csv', header=0)
# new_inout_ecms = pd.read_csv('KO8_new_inout_conversion_cone.csv', header=0)
#
# bijection_YN, ecms_first_min_ecms_second, ecms_second_min_ecms_first = check_bijection_csvs(
#     old_inout_ecms, new_inout_ecms)

pass
