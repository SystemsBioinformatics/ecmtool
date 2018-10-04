import numpy as np

## Description: model 3 with manually chosen parameters such that A = [-1 -2 4]

# Make sure each execution run has the same random numbers
np.random.seed(0)
random = np.random.random

metabolites = ['x']
metabolic_reactions = [{'x': 1}, {'x': 1}]

enzymes = ['e1', 'e2', 'r']
enzyme_reactions = [{'x': -1}, {'x': -2}, {'x': -4}, ]

metabolite_molar_volumes = [0]
enzyme_molar_volumes = [0, 0, 0]
metabolite_concentrations = {'x': 0, 'y': 0}
growth_rate = 0.2

general_kcat = 1

max_growth_rate = 0.6

# Saturation functions fi(x) (pg. 3 between eq. 10-11)
rate_functions = [lambda x: general_kcat, lambda x: general_kcat]