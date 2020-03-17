import pandas as pd
import csv
from helpers_analyze_results import *
import numpy as np
from ecmtool.network import extract_sbml_stoichiometry

model_path = 'models/iIT341.xml'

import os
import cbmpy as cbm
import csv

"""CONSTANTS"""
model_name = "iIT341"

model_dir = os.path.join(os.getcwd(), "models")
model_path = os.path.join(model_dir, model_name + ".xml")
mod = cbm.readSBML3FBC(model_path)

# Import FBA solution from file
with open('tmp/reactions_and_values_FBA_iIT341.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        if csv_reader.line_num == 1:
            re_ids = row
        else:
            re_vals = row

"""Check if efms match ecms in E. coli core"""
# Load model in ecmtool and create network object
network = extract_sbml_stoichiometry(model_path, add_objective=True,
                                     determine_inputs_outputs=True,
                                     skip_external_reactions=True)

reacs_not_in_network = []
stoichs_do_not_match = []
reversibility_wrong = []
metabolite_not_in_network = []
more_than_one_metab_in_exchange = []
direction_wrong = []
for re_ind, re_id in enumerate(re_ids):
    re_val = float(re_vals[re_ind])
    network_reaction_and_ind = [(reaction, re_ind) for re_ind, reaction in enumerate(network.reactions) if
                                reaction.id == re_id]
    if not network_reaction_and_ind:  # In this case it is an exchange reaction
        reacs_not_in_network.append(re_id)
        mod_reaction = mod.getReaction(re_id)
        cmod_stoich = mod_reaction.getStoichiometry()
        if len(cmod_stoich) > 1:
            more_than_one_metab_in_exchange.append(re_id)
        else:
            coeff = cmod_stoich[0][0]
            mod_metab = cmod_stoich[0][1]
            network_metab = [net_metab for net_metab in network.metabolites if net_metab.id == mod_metab]
            if not network_metab:
                metabolite_not_in_network.append(mod_metab)
            else:
                network_metab = network_metab[0]
                if coeff * re_val > 0.:
                    # This is used as an input
                    if network_metab.direction == 'output':
                        direction_wrong.append(mod_metab)
                else:
                    # This is used as an output
                    if network_metab.direction == 'input':
                        direction_wrong.append(mod_metab)
    else:
        network_reaction = network_reaction_and_ind[0][0]
        network_ind = network_reaction_and_ind[0][1]
        mod_reaction = mod.getReaction(re_id)
        # Check stoichiometry
        cmod_stoich = mod_reaction.getStoichiometry()
        network_stoich = [(float(network.N[metab_ind, network_ind]), metab.id) for metab_ind, metab in
                          enumerate(network.metabolites) if network.N[metab_ind, network_ind] != 0]
        stoichs_match = True
        for stoich in cmod_stoich:
            if stoich not in network_stoich:
                stoichs_match = False
        if not stoichs_match:
            stoichs_do_not_match.append(re_id)

        # Check reversibility with flux val
        if re_val < 0.:
            if not network_reaction.reversible:
                reversibility_wrong.append(re_id)

with open('tmp/essential_inputs_iIT341.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        essential_inputs = row

input_inds = []
for reac_id in essential_inputs:
    input_inds.append(
        [metab_ind for metab_ind, metab in enumerate(network.metabolites) if metab.id[1:] == reac_id[4:]][0])

print(','.join(map(str, input_inds)))
ext_inds = [metab_ind for metab_ind, metab in enumerate(network.metabolites) if metab.is_external]
output_inds = np.setdiff1d(ext_inds, input_inds)
print(','.join(map(str, output_inds)))

ex_N = network.N
identity = np.identity(len(network.metabolites))
reversibilities = [reaction.reversible for reaction in network.reactions]

# Create appropriate stoichiometric matrix N, and reversibilities vector
# Add exchange reactions so efmtool can calculate EFMs in steady state
n_exchanges = 0
reac_ids = [reac.id for reac in network.reactions]
for index, metabolite in enumerate(network.metabolites):
    if metabolite.is_external:
        reaction = identity[:, index] if metabolite.direction != 'output' else -identity[index]
        ex_N = np.append(ex_N, np.transpose([reaction]), axis=1)
        reac_ids.append('R_ex_' + metabolite.id)
        reversibilities.append(True if metabolite.direction == 'both' else False)
        n_exchanges += 1

# Store N and reversibilities to solve it with Matlab
ids = [metab.id for metab in network.metabolites]
np.savetxt('tmp/ex_N.csv', ex_N, delimiter=',')
np.savetxt('tmp/reversibilities.csv', reversibilities, delimiter=',')

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
