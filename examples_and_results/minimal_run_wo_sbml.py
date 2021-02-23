import numpy as np
from ecmtool import get_conversion_cone, get_network_from_stoich
from ecmtool.helpers import unsplit_metabolites, print_ecms_direct, unique
import pandas as pd
import os

"""Initialise stoichiometry matrix"""
# We use an example where two substrates, S1 and S2, can each be converted to X, and then X can be converted to P
stoich_mat = np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [1, 1, -1],
                       [0, 0, 1]])

"""Decide for each reaction if it is reversible"""
reversible_reac_inds = [1, 2]

"""Decide which metabolites are external (will not be held in steady-state, and will occur in ECMs)"""
ext_inds = [0, 1, 3]

"""Decide which metabolites are inputs and outputs"""
input_inds = [0, 1, 3]  # We choose to make all metabolites both inputs and outputs
output_inds = [0, 1, 3]

"""Give metabolite IDs"""
metab_names = ['S1', 'S2', 'X', 'P']

"""Get network object"""
network = get_network_from_stoich(stoich_mat, ext_inds=ext_inds, reversible_inds=reversible_reac_inds,
                                  input_inds=input_inds, output_inds=output_inds, metab_names=metab_names)

""""Split in and out metabolites. This step is essential for the correct handling of in- and output directions"""
network.split_in_out(only_rays=False)

"""Stap 2: compress network"""
network.compress(verbose=True)

"""Stap 3: Ecms enumereren"""
#  In this script, indirect intersection is used. Use command line options to use direct intersection
cone = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                           network.input_metabolite_indices(), network.output_metabolite_indices(), only_rays=False,
                           verbose=True)

cone_transpose, ids = unsplit_metabolites(np.transpose(cone), network)
cone = np.transpose(cone_transpose)

"""Print ECMs on console"""
print_ecms_direct(np.transpose(cone), ids)

"""Save ECMs as a csv-file"""
# Create dataframe with the ecms as rows and the metabolites as column headers
ecms_df = pd.DataFrame(cone, columns=ids)
ecms_df.to_csv(path_or_buf=os.path.join(os.getcwd(), 'ecms.csv'), index=True)
