import os

import cbmpy
import numpy as np
import pandas

# LB = cbmpy.readSBML3FBC('LBUL_v1_04_reducedForJulia_includingBlockedReactions.xml')
# LB = cbmpy.readSBML3FBC('kineticModel.xml')
# EXReact = LB.getReactionIds(substring='R_EX_')
#
# BoundsEX = []
# for i,React in enumerate(EXReact):
#     BoundsEX.append(LB.getReactionBounds(React))
# mod = cbmpy.readSBML3FBC('Yeast833_bigg_genesFixed_cbmpy_format.xml')
# mod = cbmpy.readCOBRASBML('Yeast833_bigg_genesFixed_cobra_format.xml')
mod = cbmpy.readSBML3FBC(os.path.join('models', 'iNF517.xml'))

# R = mod.getReactionIds('HEX1')
# Hex1 = mod.getReaction('R_HEX1')
# Hex10 = mod.getReaction('R_HEX10')
# mod.deleteReactionAndBounds('R_HEX1')
# deleteNonReactingSpecies()
# deleteReactionAndBounds()

# mod.deleteNonReactingSpecies()

DEM = cbmpy.CBTools.findDeadEndMetabolites(mod)
DER = cbmpy.CBTools.findDeadEndReactions(mod)
mod.getActiveObjective()

toRemoveKin = ['R_ENO', 'R_FBA', 'R_FBP', 'R_GAPD', 'R_LDH_L', 'R_PFK', 'R_PFL', 'R_GLCpts', 'R_PYK',
               'R_PTAr', 'R_ACKr', 'R_ALCD2x', 'R_BG_CELLB', 'R_EX_cellb_e']
MetListKin = []

for KinReact in toRemoveKin:
    react = mod.getReaction(KinReact)
    MetListKin.extend(react.getSpeciesIds())

MetListKin = list(dict.fromkeys(MetListKin))

for KinReact in toRemoveKin:
    mod.deleteReactionAndBounds(KinReact)

DelSpec = mod.deleteNonReactingSpecies(simulate=False)

MetListKin = [x for x in MetListKin if x not in DelSpec]

for met in MetListKin:
    Spec = mod.getSpecies(met)
    # Spec.setBoundary()
    mod.createReaction('R_EX_' + met, name=met + 'Exchange Reaction')
    nEXr = mod.getReaction('R_EX_' + met)
    nEXr.createReagent(met, coefficient=-1.0)

# DEMred = cbmpy.CBTools.findDeadEndMetabolites(mod)
DERred = cbmpy.CBTools.findDeadEndReactions(mod)  # Note that there might be metabolites with two exchange reactions now

EXreact = cbmpy.CBTools.processExchangeReactions(mod, 'R_EX_')
# cbmpy.CBTools.checkExchangeReactions(mod, autocorrect=True)

MetListEX = []
for EXr in EXreact[0]:
    react = mod.getReaction(EXr)
    MetListEX.extend(react.getSpeciesIds())

for EXr in EXreact[1]:
    react = mod.getReaction(EXr)
    MetListEX.extend(react.getSpeciesIds())

MetListEX = list(dict.fromkeys(MetListEX))
df_MetListEX = pandas.DataFrame(MetListEX)  # Note that there are some doubles here too

full_EXreact = {**EXreact[0], **EXreact[1]}
input_MetList = []
output_MetList = []
both_MetList = []
for EXr in full_EXreact:
    react = mod.getReaction(EXr)
    if full_EXreact[EXr]['lb'] < 0:  # This is an input or a both
        if full_EXreact[EXr]['ub'] <= 0:  # This is an input
            input_MetList.extend(react.getSpeciesIds())
        else:  # This is a both
            both_MetList.extend(react.getSpeciesIds())
    else:  # This is an output
        output_MetList.extend(react.getSpeciesIds())

non_dupe_input_MetList=[]
non_dupe_output_MetList=[]
non_dupe_both_MetList=[]
for metab in input_MetList + both_MetList + output_MetList:
    if metab in input_MetList:
        if metab not in both_MetList + output_MetList:
            if metab not in non_dupe_input_MetList:
                non_dupe_input_MetList.append(metab)
        elif metab not in non_dupe_both_MetList:
            non_dupe_both_MetList.append(metab)
    elif metab in output_MetList:
        if metab not in both_MetList:
            if metab not in non_dupe_output_MetList:
                non_dupe_output_MetList.append(metab)
        elif metab not in non_dupe_both_MetList:
            non_dupe_both_MetList.append(metab)
    else:
        if metab not in non_dupe_both_MetList:
            non_dupe_both_MetList.append(metab)

df_EXreact_output = pandas.DataFrame(non_dupe_output_MetList)
df_EXreact_output['direction'] = 'output'
df_EXreact_input = pandas.DataFrame(non_dupe_input_MetList)
df_EXreact_input['direction'] = 'input'
df_EXreact_both = pandas.DataFrame(non_dupe_both_MetList)
df_EXreact_both['direction'] = 'both'

df_EXreact_directions = pandas.concat([df_EXreact_input,df_EXreact_output, df_EXreact_both])
hide_list = [metab not in MetListKin for metab in df_EXreact_directions[0]]
df_EXreact_directions['hide'] = hide_list

for irR in mod.getReactionIds():  # Get rid of reactions with minimal velocities
    Bounds = mod.getReactionBounds(irR)
    if Bounds[1] > 0:
        # print(irR)
        # print(Bounds)
        mod.setReactionLowerBound(irR, 0)
        mod.setReactionUpperBound(irR, 1000)
    if Bounds[2] < 0:
        # print(irR)
        # print(Bounds)
        mod.setReactionLowerBound(irR, -1000)
        mod.setReactionUpperBound(irR, 0)

# cbmpy.CBWrite.writeSBML3FBC(mod, 'LactisFULLWOGlycolysis.xml')
solFBA = cbmpy.doFBA(mod)
solFVA = cbmpy.doFVA(mod)

a = solFVA[0][:, [2, 3]]
indZeroFluxOpt = np.where(~a.any(axis=1))[0]

for ind in indZeroFluxOpt:
    mod.setReactionBounds(solFVA[1][ind], 0, 0)

cbmpy.doFBA(mod)

InconsEX = cbmpy.CBTools.checkExchangeReactions(mod, autocorrect=False)
InconsBounds = cbmpy.CBTools.checkFluxBoundConsistency(mod)

cbmpy.CBWrite.writeSBML3FBC(mod, os.path.join('models','LactisFULLWOGlycolysis.xml'))

"""Find FBA-conversion"""
fba_dict = {}
for metab in df_EXreact_directions[0]:
    fba_dict.update({metab: 0})

for metab_reac_tuple in DERred:
    m_id = metab_reac_tuple[0]
    r_id = metab_reac_tuple[1]
    reac = mod.getReaction(r_id)
    val = reac.getValue()
    stoich_coeff = reac.reagents[0].coefficient
    metab_prod = - val * stoich_coeff
    fba_dict[m_id] += metab_prod

df_EXreact_directions['FBA_result'] = 0
for ind, metab in enumerate(df_EXreact_directions[0]):
    df_EXreact_directions['FBA_result'].values[ind] = fba_dict[metab]

df_EXreact_directions.to_csv("df_EXreact_Lactis_directions.csv", sep=',', index=False)
# cbmpy.CBWrite.writeSBML3FBC(mod, 'LactisFULLECM.xml')

# import pysces

# pysces.interface.convertSBML2PSC('Ecoli_Chassagnole_org.xml', sbmldir='C:\Users\JNR520\surfdrive\Python\Ikin\yeast', pscfile='Ecoli_Chassagnole_org.psc', pscdir='C:\Users\JNR520\surfdrive\Python\Ikin\yeast')
# modKin = pysces.model('C:\Users\JNR520\surfdrive\Python\Ikin\yeast\Ecoli_Chassagnole_org.psc')

print('end')
