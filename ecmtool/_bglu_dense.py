import sys

def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, imp
    if sys.platform == "linux" or sys.platform == "linux2":
        __file__ = pkg_resources.resource_filename(__name__, 'bglu_dense.cpython-37m-x86_64-linux-gnu.so')
    elif sys.platform == "darwin" :
        __file__ = pkg_resources.resource_filename(__name__, 'bglu_dense.cpython-37m-darwin.so')
    elif sys.platform == 'windows':
        raise EnvironmentError('Windows is not supported for BGLU functionality, please use Linux or Mac OS for now.')
    else:
        raise EnvironmentError('Unknown operating system, we don\'t know what BLGU binary to load.')
    __loader__ = None; del __bootstrap__, __loader__
    imp.load_dynamic(__name__,__file__)
__bootstrap__()

