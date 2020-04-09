import os
import cbmpy as cbm
import csv
from results_analysis.helpers_ECM_calc_lactis_wo_glycolysis import *

"""CONSTANTS"""
model_name = "LactisFULLWOGlycolysis"

# Define directories for finding models
model_dir = os.path.join(os.getcwd(), "models")

model_path = os.path.join(model_dir, model_name + ".xml")
mod = cbm.readSBML3FBC(model_path)

# minimal medium seems to be
# <class 'list'>: ['Sink reaction for O2', 'Sink reaction for L-glutamine', 'Sink reaction for myo-inositol',
# 'Sink reaction for N2', 'Sink reaction for sucrose']

ecms_matrix, full_network_ecm = calc_ECMs(model_path, print_results=True)

# Find all metabolite_ids in the order used by ECMtool
metab_ids = [metab.id for metab in full_network_ecm.metabolites]
# Create dataframe with the ecms as columns and the metabolites as index
ecms_df = pd.DataFrame(ecms_matrix, index=metab_ids)

ext_indices = [ind for ind, metab in enumerate(full_network_ecm.metabolites) if metab.is_external]
ext_metab_ids = [metab for ind, metab in enumerate(metab_ids) if ind in ext_indices]
ecms_matrix_ext = ecms_matrix[ext_indices, :]
ecms_df = pd.DataFrame(ecms_matrix_ext, index=ext_metab_ids)

ecms_df.to_csv(path_or_buf='ecms_iJR904_ext.csv', index=True)
ecms_df.to_csv(path_or_buf='ecms_iJR904.csv', index=True)
