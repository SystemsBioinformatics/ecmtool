import numpy as np

# Make sure each execution run has the same random numbers
np.random.seed(0)
random = np.random.random

metabolites = ['x', 'y']
metabolic_reactions = [{'x': 1, 'y': 0}, {'x': -1, 'y': 1}, {'x': -1, 'y': 3}]

enzymes = ['e1', 'e2', 'e3', 'r']
enzyme_reactions = [{'y': -1}, {'y': -2}, {'y': -2}, {'y': -4}, ]

metabolite_molar_volumes = [0, 0]
enzyme_molar_volumes = [random(), random(), random(), random()]
metabolite_concentrations = {'x': 0, 'y': 0}
enzyme_concentrations = {'e1': random(), 'e2': random(), 'e3': random()}
growth_rate = random()

# Saturation functions fi(x) (pg. 3 between eq. 10-11)
rate_functions = [lambda x: (x['e1']), lambda x: (x['x'] * x['e2']), lambda x: (x['x'] * x['e3'])]