import numpy
import libsbml
import cbmpy
from cbmpy import CBSolver


if __name__ == '__main__':
    mod = cbmpy.readSBML3FBC('models/e_coli_core.xml')
    fba = cbmpy.doFBA(mod)
    pass