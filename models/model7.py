import numpy as np

## Description: new model with 3 unique paths to ribosome activity

# Make sure each execution run has the same random numbers
np.random.seed(0)
uniform = np.random.uniform

metabolites = ['x', 'y']
metabolic_reactions = [{'x': 1, 'y': 0}, {'x': 0, 'y': 1}, {'x': -1, 'y': 1}]

enzymes = ['e1', 'e2', 'e3', 'r']
enzyme_reactions = [{'y': -3}, {'y': -2}, {'y': -1}, {'y': -4}]

metabolite_molar_volumes = [uniform(10**-6, 10**-1), uniform(10**-6, 10**-1)]
enzyme_molar_volumes = [uniform(10**-6, 10**-1), uniform(10**-6, 10**-1), uniform(10**-6, 10**-1), uniform(10**-6, 10**-1)]
metabolite_concentrations = {'x': 0, 'y': 0}

kcat_e1 = 0.5
kcat_e2 = 0.7
kcat_e3 = 0.3

max_growth_rate = 0.4

# Saturation functions fi(x) (pg. 3 between eq. 10-11)
rate_functions = [lambda x: kcat_e1, lambda x: kcat_e2, lambda x: kcat_e3]