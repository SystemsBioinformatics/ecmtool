import numpy as np

# Make sure each execution run has the same random numbers
np.random.seed(0)
random = np.random.random

metabolites = ['x', 'y']
metabolic_reactions = [{'x': 1}, {'x': -1, 'y': 1}, {'y': 1}]

enzymes = ['e1', 'e2', 'e3', 'r']
enzyme_reactions = [{'x': -3}, {'x': -2}, {'x': -1}, {'x': -4}, ]

metabolite_molar_volumes = [0, 0]
enzyme_molar_volumes = [0, 0, 0, 0]
metabolite_concentrations = {'x': 0, 'y': 0}
enzyme_concentrations = {'e1': 1, 'e2': 0.5}
growth_rate = 0.5

# Saturation functions fi(x) (pg. 3 between eq. 10-11)
rate_functions = [lambda x: (x['e1']), lambda x: (x['e2'])]