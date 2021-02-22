# ecmtool - Uncover organisms' metabolic blueprints

With this tool you can calculate _Elementary Conversion Modes_ (ECMs) from metabolic networks. Combinations of ECMs form all metabolic influence an organism can exert on its environment.

ecmtool can be used in two different modes: either as a standalone command line tool, or as a Python library for your own scripts. We will describe how to install and use both modes.

### Prerequisites
* Download and install Python. Ecmtool is compatible with python 3.x. Ensure both python and its package manager _pip_ are added to your PATH environment variable. If this last step is omitted, an error like the following will be thrown when you try to run python: `’python’ is not recognized as an internal or external command [..]`.
* Download and install Java.

## Mode 1: standalone command line tool
In this mode, you can call ecmtool like a normal program from your command line. It reads metabolic networks in the SBML format, and writes resulting ECMs into a CSV file for later analysis. Most researchers will use this method. For running ecmtool on computing clusters efficiently, see the Advanced Usage section in this readme.

### Installation
* Download the latest ecmtool source through `git clone`, or as a zip file from https://github.com/tjclement/ecmtool.
* Open a command prompt, and navigate to the ecmtool directory (e.g. `cd C:\Users\You\Git\ecmtool`, where the
path should be replaced with the path ecmtool was downloaded to).
* Install the dependencies in requirements.txt inside the ecmtool directory (e.g. by running `pip install -r requirements.txt`).
* Linux only: install _redund_ of package _lrslib_ (e.g. by running `apt install lrslib`).

### Running
Ecmtool can be run by executing `python main.py –-model_path <path/to/model.xml> [arguments]` from the command line, after navigating to the ecmtool directory as described above. The possible arguments and their default values are printed when you run `python main.py --help`.
After execution is done, the found conversions have been written to file (default: _conversions.csv_). The first row of this CSV file contain the metabolite IDs as read from the SBML model.

### Example

```bash
python main.py --model_path models/e_coli_core.xml --auto_direction true --out_path core_conversions.csv
```

## Mode 2: Python library
ecmtool can also be used as a separate programming interface from within your own Python code. To do so, install ecmtool using _pip_ (e.g. `pip install ecmtool`). The most crucial method is ecmtool.conversion_cone:get_conversion_cone(), which returns the ECMs of a given stoichiometric matrix. For information on how to use advanced features like SBML parsing, network compression, and metabolite direction estimation, please see ecmtool/main.py.


### Example
```python
from ecmtool.network import extract_sbml_stoichiometry
from ecmtool.conversion_cone import get_conversion_cone

network = extract_sbml_stoichiometry('models/sxp_toy.xml', add_objective=True)
stoichiometry = network.N

ecms = get_conversion_cone(stoichiometry, network.external_metabolite_indices(),
 network.reversible_reaction_indices(), network.input_metabolite_indices(), 
 network.output_metabolite_indices())
```

## Advanced usage
After testing how the tool works, most users will want to run their workloads on computing clusters instead of on single machines. This section describes some of the steps that are useful for running on clusers

### Parallel computing with mpi4py
On Linux or Mac, ecmtool can make use of mpi4py for parallel computing. To make use of this feature, an implementation of MPI (e.g. OpenMPI) and mpi4py are required. They can be installed, for example, with

```
apt install openmpi-bin
pip3 install mpi4py
```

### Doubling direct enumeration method speed
The direct enumeration method can be sped up by compiling our LU decomposition code with Cython. The following describes the steps needed on Linux, but the same concept also applies to Mac OS and Windows. First make sure all dependencies are satisfied. Then execute:

```
python3 cython_setup.py build_ext --inplace

mv _bglu* ecmtool/
```


### Running on a computing cluster with mpiexec
For example: mpiexec -n 4 python3 main.py --model_path models/e_coli_core.xml


## Citing ecmtool
Please refer to the following paper when using ecmtool: [https://www.cell.com/patterns/fulltext/S2666-3899(20)30241-5](https://www.cell.com/patterns/fulltext/S2666-3899(20)30241-5).


## Acknowledgements
The original source code with indirect enumeration was written by Tom Clement. Erik Baalhuis later expanded the code with a direct enumeration method that improved parallellisation. Daan de Groot helped with many new features, bug fixes, and code reviews.

## License

ecmtool is released with the liberal MIT license. You are free to use it for any purpose. We hope others will contribute to the field by making derived work publicly available too.
