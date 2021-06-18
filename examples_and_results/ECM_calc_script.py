import os
import cbmpy as cbm
from examples_and_results.helpers_ECM_calc import *

"""CONSTANTS"""
model_name = "bacteroid"
input_file_name = "bacteroid_ECMinputSmaller.csv"
# For a bigger computation, try:
# input_file_name = "bacteroid_ECMinputAll.csv"

# Define directories for finding models
model_dir = os.path.join(os.getcwd(), "models")

model_path = os.path.join(model_dir, model_name + ".xml")
mod = cbm.readSBML3FBC(model_path)
input_file_path = os.path.join(os.getcwd(), "input_files", input_file_name)

ecms_matrix, metab_ids, full_network = calc_ECMs(model_path, print_results=True, input_file_path=input_file_path)

# Create dataframe with the ecms as columns and the metabolites as index
ecms_df = pd.DataFrame(np.transpose(ecms_matrix), columns=metab_ids)

ext_metab_inds = []
ext_metab_ids = []
for ind_id, metab_id in enumerate(metab_ids):
    ext_metab_bool = [metab.is_external for ind, metab in enumerate(full_network.metabolites) if metab.id == metab_id]
    if len(ext_metab_bool) > 0 and ext_metab_bool[0]:
        ext_metab_inds.append(ind_id)
        ext_metab_ids.append(metab_id)

ecms_matrix_ext = ecms_matrix[ext_metab_inds, :]
ecms_df_ext = pd.DataFrame(np.transpose(ecms_matrix_ext), columns=ext_metab_ids)

ecms_df_ext.to_csv(path_or_buf=os.path.join(os.getcwd(), "result_files", 'ecms_bacteroid_ext.csv'), index=True)
ecms_df.to_csv(path_or_buf=os.path.join(os.getcwd(), "result_files", 'ecms_bacteroid.csv'), index=True)
