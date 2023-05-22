# ecmtool - Uncover organisms' metabolic blueprints

With this tool you can calculate _Elementary Conversion Modes_ (ECMs) from metabolic networks. Combinations of ECMs comprise all metabolic influences an organism can exert on its environment.

ecmtool can be used in two different modes: either as a standalone command line tool, or as a Python library for your own scripts. We will describe how to install and use both modes.

### Prerequisites
* Download and install Python. Ecmtool is compatible with python 3.x, and tested on 3.10. Ensure both python and its package manager _pip_ are added to your PATH environment variable. If this last step is omitted, an error like the following will be thrown when you try to run python: `’python’ is not recognized as an internal or external command [..]`.
* Download and install Java. ecmtool is tested with OpenJDK 17. Make sure you have a 64bit version; you can check this with `java -version`. Otherwise, you might get an error `Invalid maximum heap size`.

## Mode 1: standalone command line tool
In this mode, you can call ecmtool like a normal program from your command line. It reads metabolic networks in the SBML format, and writes resulting ECMs into a CSV file for later analysis. Most researchers will use this method. For running ecmtool on computing clusters efficiently, see the Advanced Usage section in this readme.

### Installation
* Download the latest ecmtool source through `git clone`, or as a zip file from https://github.com/tjclement/ecmtool.
* Open a command prompt, and navigate to the ecmtool directory (e.g. `cd C:\Users\You\Git\ecmtool`, where the
path should be replaced with the path ecmtool was downloaded to).
* Install the dependencies in requirements.txt inside the ecmtool directory (e.g. by running `pip install -r requirements.txt`).
* Linux only: install _redund_ of package _lrslib_ (e.g. by running `apt install lrslib`).

#### Installing ecmtool using Docker 
For convenience, there's a Docker script you can use that has all dependencies already installed, and allows you to directly run ecmtool.
Open a terminal with the ecmtool project as its working directory, and run:

```bash
docker build -t ecmtool -f docker/Dockerfile .
docker run -ti ecmtool bash
```

#### Installing ecmtool using Singularity
To be continued.

### Running
Ecmtool can be run by executing `python main.py –-model_path <path/to/model.xml> [arguments]` from the command line, after navigating to the ecmtool directory as described above. The possible arguments and their default values are printed when you run `python main.py --help`.
After execution is done, the found conversions have been written to file (default: _conversions.csv_). The first row of this CSV file contain the metabolite IDs as read from the SBML model.

### Example

```bash
python main.py --model_path models/e_coli_core.xml --auto_direction true --out_path core_conversions.csv
```

### Benefiting from optional arguments of ecmtool
For an elaborate discussion of all optional arguments that can be used when ecmtool is run as a command line tool, please see the extensive manual that was uploaded as a Supplementary File with the ecmtool-publication at: https://doi.org/10.1016/j.patter.2020.100177

## Mode 2: Python library
ecmtool can also be used as a separate programming interface from within your own Python code. To do so, install ecmtool using _pip_ (e.g. `pip install ecmtool`). The most crucial method is ecmtool.conversion_cone:get_conversion_cone(), which returns the ECMs of a given stoichiometric matrix. For information on how to use advanced features like SBML parsing, network compression, and metabolite direction estimation, please see ecmtool/main.py.

*We strongly advise the user to either use ecmtool as a command line tool, or to pay much attention to carefully copy the order from ecmtool/main.py.*


### Example
```python
from ecmtool.network import extract_sbml_stoichiometry
from ecmtool.conversion_cone import get_conversion_cone
from ecmtool.helpers import unsplit_metabolites, print_ecms_direct
import numpy as np

DETERMINE_INPUTS_OUTPUTS = False # Determines whether ecmtool tries to infer directionality (input/output/both)
PRINT_CONVERSIONS = True # Prints the resulting ECMs on the console

network = extract_sbml_stoichiometry('models/sxp_toy.xml', add_objective=True, determine_inputs_outputs=DETERMINE_INPUTS_OUTPUTS)

# Some steps of compression only work when cone is in one orthant, so we need to split external metabolites with
# direction "both" into two metabolites, one of which is output, and one is input
network.split_in_out(only_rays=False)

# It is generally a good idea to compress the network before computation
network.compress(verbose=True, SCEI=True, cycle_removal=True, remove_infeasible=True)

stoichiometry = network.N

ecms = get_conversion_cone(stoichiometry, network.external_metabolite_indices(),
 network.reversible_reaction_indices(), network.input_metabolite_indices(), 
 network.output_metabolite_indices(), verbose=True)
 
# Since we have split the "both" metabolites, we now need to unsplit them again
cone_transpose, ids = unsplit_metabolites(np.transpose(ecms), network)
cone = np.transpose(cone_transpose)

# We can remove all internal metabolites, since their values are zero in the conversions (by definition of internal)
internal_ids = []
for metab in network.metabolites:
    if not metab.is_external:
        id_ind = [ind for ind, id in enumerate(ids) if id == metab.id]
        if len(id_ind):
            internal_ids.append(id_ind[0])

ids = list(np.delete(ids, internal_ids))
cone = np.delete(cone, internal_ids, axis=1)

# If you wish, one can print the ECM results:
if PRINT_CONVERSIONS:
    print_ecms_direct(np.transpose(cone), ids)

```

### Example scripts
See the scripts in the folder examples_and_results for examples on how to use ecmtool as a library. In particular: ECM_calc_script.py, compare_efms_ecms_number.py.

### Enumerating ECMs without an SBML-file 
See the script examples_and_results/minimal_run_wo_sbml.py for an example on how to compute ECMs starting from a stoichiometric matrix, and some additional information.


## Advanced usage
After testing how the tool works, most users will want to run their workloads on computing clusters instead of on single machines. This section describes some of the steps that are useful for running on clusers

### Parallel computing with OpenMPI
On Linux or Mac, ecmtool can make use of OpenMPI for running on parallel in a computing cluster. To make use of this feature, in addition to the dependencies in requirements.txt, OpenMPI, mpi4py, and mplrs are required. The installation of OpenMPI and mplrs is done via:

```bash
apt install libopenmpi-dev
wget http://cgm.cs.mcgill.ca/~avis/C/lrslib/archive/lrslib-071a.tar.gz
tar -xzf lrslib-071a.tar.gz
cd lrslib-071a
make && make mplrs && make install
ln -s `pwd`/mplrs /usr/local/bin/mplrs
ln -s `pwd`/redund /usr/local/bin/redund
cd ..
```

The installation of mpi4py is done via:
```bash
pip3 install mpi4py==3.1.4
```

Running ecmtool on a cluster using the indirect enumeration method is now as simple as running:
```bash
python3 main.py --processes <number of processes for enumeration> --model_path models/e_coli_core.xml
```
Note that this performs preprocessing steps like network compression on the node you run this command on, and not on the compute cluster.

For direct enumeration, the number of processes for enumeration is passed to mpiexec instead:
```bash
mpiexec -n <number of processes for enumeration> python3 main.py --direct true --model_path models/e_coli_core.xml
```
In this mode, preprocessing steps are run on the compute cluster too.

### Advanced ECM-computation on a computing cluster
#### Installation of ecmtool when the user does not have root privileges on the cluster (a case report)
On some computing clusters, it is not easy to install OpenMPI and mplrs. One method that was successful is outlined here. This cluster had an OpenMPI already available as a module that could be loaded. The available versions can be seen by 
```bash
module av OpenMPI
```
For the installation of mplrs, we will also need GMP, check this by
```bash
module av GMP
```
It is important that the versions of OpenMPI and GMP have to match. In this case, we used
```bash
module load OpenMPI/4.1.1-GCC-10.3.0
module load GMP/6.2.1-GCCcore-10.3.0
```
where the last number indicates that they are using a compatible version of GCC. Now, we are ready to install mplrs. This can be done via:
```bash
apt install libopenmpi-dev
wget http://cgm.cs.mcgill.ca/~avis/C/lrslib/archive/lrslib-071a.tar.gz
tar -xzf lrslib-071a.tar.gz
cd lrslib-071a
make && make mplrs && make install
```
Now we need to tell the cluster where to find the installed mplrs. We can do this by adding the path to mplrs to the search path:
```bash
export LD_LIBRARY_PATH=/scicore/home/nimwegen/degroo0000/ecmtool/lrslib-071a:$LD_LIBRARY_PATH
export PATH=/scicore/home/nimwegen/degroo0000/ecmtool/lrslib-071a:$PATH
```
Now using the command
```bash
mplrs
```
should give some output that indicates that mplrs is working and can be found.

#### Running ecmtool using separate runs for non-parallel and parallel parts, with a .sh-script (on a slurm-cluster)
To fully exploit parallel computation on a cluster, one would like to use ecmtool in separate steps, as outlined below. (In the ecmtool-folder one can also find an example-script that can be used on a computing cluster that is using slurm: ```examples_and_results/launch_separate_mmsyn_newest.sh```.)

1. preprocessing and compression of the model on a compute node (instead of a login node). For this run 
```bash
srun --ntasks=1 --nodes=1 python3 main.py all_until_mplrs --model_path ${MODEL_PATH} --auto_direction ${AUTO_DIRECT} --hide "${HIDE}" --prohibit "${PROHIBIT}" --tag "${TAG}" --inputs "${INPUTS}" --outputs "${OUTPUTS}" --use_external_compartment "${EXT_COMP}" --add_objective_metabolite "${ADD_OBJ}" --compress "${COMPRESS}" --hide_all_in_or_outputs "${HIDE_ALL_IN_OR_OUTPUTS}
```
where the arguments in curly brackets should be replaced by your choices for these arguments.

2. first vertex enumeration step with mplrs in parallel. For this run 
```bash
mpirun -np <number of processes> mplrs -redund ecmtool/tmp/mplrs.ine ecmtool/tmp/redund.ine
mpirun -np <number of processes>  mplrs ecmtool/tmp/redund.ine ecmtool/tmp/mplrs.out
```

3. processing of results from first vertex enumeration step, adding steady-state constraints and removing redundant rays using a parallelized redundancy check.
```bash
mpirun -np <number of processes> python3 main.py all_between_mplrs
```

4. second vertex enumeration step with mplrs in parallel
```bash
mpirun -np <number of processes> mplrs -redund ecmtool/tmp/mplrs.ine ecmtool/tmp/redund.ine
mpirun -np <number of processes>  mplrs ecmtool/tmp/redund.ine ecmtool/tmp/mplrs.out
```

5. processing of results from second vertex enumeration step, unsplitting of metabolites, ensuring that results are unique, and saving ecms to file
```bash
srun --ntasks=1 --nodes=1 python3 main.py all_from_mplrs --out_path ${OUT_PATH}
```

### Doubling direct enumeration method speed
The direct enumeration method can be sped up by compiling our LU decomposition code with Cython. The following describes the steps needed on Linux, but the same concept also applies to Mac OS and Windows. First make sure all dependencies are satisfied. Then execute:

```
python3 cython_setup.py build_ext --inplace

mv _bglu* ecmtool/
```

ℹ️ Note that in the Docker script, this optimisation has already been done. You don't need to compile anything there.

## Automatically testing ecmtool and contributing to ecmtool
When ecmtool is installed properly its functioning with various parameter settings can be tested using some predefined tests using
```bash
python3 -m pytest tests/test_conversions.py
```
When contributing to ecmtool please make sure that these tests are passed before making a pull request. 

## Citing ecmtool
Please refer to the following papers when using ecmtool:

Initial version - [https://www.cell.com/patterns/fulltext/S2666-3899(20)30241-5](https://www.cell.com/patterns/fulltext/S2666-3899(20)30241-5).

`mplrs` improved version - [https://doi.org/10.1093/bioinformatics/btad095](https://doi.org/10.1093/bioinformatics/btad095).
`mplrs`-improved version - [https://doi.org/10.1093/bioinformatics/btad095](https://doi.org/10.1093/bioinformatics/btad095).

## Acknowledgements
The original source code with indirect enumeration was written by [Tom Clement](https://scholar.google.com/citations?user=kUD5y04AAAAJ). [Erik Baalhuis](https://github.com/EBaalhuis) later expanded the code with a direct enumeration method that improved parallellisation. [Daan de Groot](https://scholar.google.com/citations?user=xY_GjWkAAAAJ) helped with many new features, bug fixes, and code reviews. [Bianca Buchner](https://github.com/BeeAnka) added support for `mplrs`, which raises the maximal size of networks you can enumerate with ecmtool. 

## License
ecmtool is released with the liberal MIT license. You are free to use it for any purpose. We hope others will contribute to the field by making derived work publicly available too.
