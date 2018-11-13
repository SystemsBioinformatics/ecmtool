import numpy
import libsbml
import cbmpy


def do_fba(filename):
    mod = cbmpy.readSBML3FBC(filename)
    cbmpy.doFBA(mod)
    return mod.getReactionValues()


if __name__ == '__main__':
    result = do_fba('models/e_coli_core_constr.xml')
    pass