import cbmpy as cbm

from examples_and_results.helpers_ECM_calc import *

"""CONSTANTS"""
model_name = "iND750"
# For a bigger computation, try:
# input_file_name = "bacteroid_ECMinputAll.csv"

# Define directories for finding models
model_dir = os.path.join(os.getcwd(), "models")

model_path = os.path.join(model_dir, model_name + ".xml")
mod = cbm.readSBML3FBC(model_path)

#
pairs = cbm.CBTools.findDeadEndReactions(mod)
external_metabolites, external_reactions = list(zip(*pairs)) if len(pairs) else (
    list(zip(*cbm.CBTools.findDeadEndMetabolites(mod)))[0], [])

# External according to Urbanczik
ext_urbanczik = ['ac', 'acald', 'ala__L', 'co2', 'csn', 'ergst', 'etoh', 'gam6p', 'glc__D', 'hdcea', 'ocdcea', 'ocdcya',
                 'so4', 'xylt', 'zymst', 'nh4', 'asp__L', 'ser__L', 'fum', 'gly', 'thr__L']
force_feed = ['ac']

ext_urbanczik_inds = [ind for ind, metab in enumerate(external_metabolites) if metab[2:-2] in ext_urbanczik]
force_urbanczik_inds = [ind for ind, metab in enumerate(external_metabolites) if metab[2:-2] in force_feed]
for ind, re_id in enumerate(external_reactions):
    if ind in ext_urbanczik_inds:
        mod.setReactionBounds(re_id, -10, 10)
    else:
        mod.setReactionBounds(re_id, 0, 0)
    if ind in force_urbanczik_inds:
        mod.setReactionBounds(re_id, 0.1, 0.1)



# Get exchange reactions
exch_inds = [(ind, reac.id) for ind, reac in enumerate(mod.reactions) if reac.id[:4] == 'R_EX']
medium_list = []
