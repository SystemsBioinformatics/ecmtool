from setuptools import setup
import numpy as np
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='ecmtool',
    version='0.1.5',
    packages=['ecmtool'],
    install_requires=['numpy',
                      'scipy',
                      'sympy',
                      'python-libsbml',
                      'cbmpy',
                      'pycddlib',
                      'psutil',
                      'sklearn'],
    ext_modules=cythonize(Extension("_bglu_dense", ["ecmtool/_bglu_dense.pyx"], include_dirs=[np.get_include()]), compiler_directives={'language_level' : "3"}),
    include_dirs=[np.get_include()],
    url='https://github.com/tjclement/ecmtool',
    license='MIT',
    author='Tom Clement',
    author_email='mr@tomclement.nl',
    description='Calculates elementary conversion modes (Urbanczik & Wagner, 2005) of metabolic networks.',
    include_package_data=True
)
