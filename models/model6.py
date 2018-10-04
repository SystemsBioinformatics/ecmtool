import numpy as np

## Description: model 5 with modified kcat2 such that EGM asymptotes of B1 and B2 do not share their B3 across EGMs

# Make sure each execution run has the same random numbers
np.random.seed(0)
uniform = np.random.uniform

metabolites = ['x']
metabolic_reactions = [{'x': 1}, {'x': 1}]

enzymes = ['e1', 'e2', 'r']
enzyme_reactions = [{'x': -1}, {'x': -2}, {'x': -4}, ]

metabolite_molar_volumes = [uniform(10**-6, 10**-1)]
enzyme_molar_volumes = [uniform(10**-6, 10**-1), uniform(10**-6, 10**-1), uniform(10**-6, 10**-1)]
metabolite_concentrations = {'x': 0, 'y': 0}

kcat_e1 = 0.5
kcat_e2 = 0.8

max_growth_rate = 0.6

# Saturation functions fi(x) (pg. 3 between eq. 10-11)
rate_functions = [lambda x: kcat_e1, lambda x: kcat_e2]