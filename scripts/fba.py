import numpy
import libsbml
import cbmpy


def do_fba(filename):
    mod = cbmpy.readSBML3FBC(filename)
    cbmpy.doFBAMinSum(mod)
    # cbmpy.doFBA(mod)
    return mod.getReactionValues(), mod.getObjFuncValue()


if __name__ == '__main__':
    result, objective = do_fba('models/iNF517.xml')
    pass