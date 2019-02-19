from setuptools import setup

setup(
    name='ecmtool',
    version='0.1.4',
    packages=['ecmtool'],
    install_requires=['numpy',
                      'scipy',
                      'sympy',
                      'python-libsbml',
                      'cbmpy',
                      'pycddlib',
                      'psutil'],
    url='https://github.com/tjclement/ecmtool',
    license='MIT',
    author='Tom Clement',
    author_email='mr@tomclement.nl',
    description='Calculates elementary conversion modes (Urbanczik & Wagner, 2005) of metabolic networks.',
    include_package_data=True
)
