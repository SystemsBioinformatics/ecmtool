from setuptools import setup
import numpy as np
from distutils.extension import Extension

setup(
    name='ecmtool',
    version='0.1.9',
    packages=['ecmtool'],
    install_requires=['numpy',
                      'scipy',
                      'sympy',
                      'python-libsbml',
                      'cbmpy',
                      'psutil',
                      'sklearn'],
    include_dirs=[np.get_include()],
    url='https://github.com/SystemsBioinformatics/ecmtool',
    license='MIT',
    author='Tom Clement, Erik Baalhuis, Daan de Groot',
    author_email='science@tomclement.nl',
    description='Calculates elementary conversion modes (Urbanczik & Wagner, 2005) of metabolic networks.',
    include_package_data=True
)
