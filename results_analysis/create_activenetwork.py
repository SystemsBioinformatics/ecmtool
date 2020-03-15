import numpy as np
import os
import cbmpy as cbm

"""CONSTANTS"""
model_name = "iIT341"

model_dir = os.path.join(os.getcwd(), "models")
model_path = os.path.join(model_dir, model_name + ".xml")
original_cmod = cbm.readSBML3FBC(model_path)
cbm.doFBA(original_cmod)

delete_reaction = []  # Initializes list for reactions that need to be deleted
opt_obj = original_cmod.getObjFuncValue()  # Needed for checking if we did not remove too much
zero_tol = 1e-9
opt_tol = 1e-8
zero_tol = zero_tol * opt_obj
deletion_success = False
counter = 0
N_ITER = 10
cmod = original_cmod.clone()

while not deletion_success:  # Runs until we have thrown away only reactions that do not affect the objective
    for rid in cmod.getReactionIds():
        reaction = cmod.getReaction(rid)
        if abs(reaction.value) <= zero_tol:
            delete_reaction.append(rid)

    # delete all reactions in delete_reaction from model
    for rid in delete_reaction:
        print(rid)
        cmod.deleteReactionAndBounds(rid)

    cbm.doFBA(cmod)
    changed_obj = abs(cmod.getObjFuncValue() - opt_obj)
    if changed_obj <= opt_tol:  # Check if objective value didn't change
        deletion_success = True
    else:
        cmod = original_cmod
        zero_tol = zero_tol / 10  # If we have removed to much: Adjust zero_tol and try again

    if counter <= N_ITER:
        counter += 1
    else:
        print("Reaction deletion did not succeed within %d iterations." % N_ITER)

# Then delete metabolites that are no longer used by any reaction
stoich_matrix = cmod.N.array
n_reacs_per_metab = np.count_nonzero(stoich_matrix, axis=1)
inactive_metabs = [cmod.species[metab_index].id for metab_index in range(stoich_matrix.shape[0]) if
                   n_reacs_per_metab[metab_index] == 0]

for metab_id in inactive_metabs:
    print('Deleting %s because it is not used in active network.' % metab_id)
    cmod.deleteSpecies(metab_id)

model_path = os.path.join(model_dir, 'iIT341_active' + ".xml")

cbm.writeSBML3FBCV2(cmod, model_path, add_cbmpy_annot=False)

