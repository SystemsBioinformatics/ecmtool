### Readme for the examples_and_results folder

In this folder, some example scripts have been gathered. These scripts show how ecmtool can be run by using a script, rather than by using the command line. A short description of the scripts

### `compare_efms_ecms_number.py`

In this script, for several models the number of EFMs and ECMs is calculated and compared. Note that this script only runs when Matlab and EFMtool (https://csb.ethz.ch/tools/software/efmtool.html) are installed. Output of this script is stored in the folder `result_files`

### `ECM_calc_script.py`

This script calculates ECMs for a model that can be given in at the start. As an example, we calculate the ECMs for a model of a rhizobial bacteroid created by Carolin Schulte, but SBML-file can be used, by giving the file_path to the model in the varaible `model_path`. In the variable `input_file_path`, the user can give the link to a file that gives information for all external metabolite. To be precise, if the metabolite can be used as an input, output, and if the metabolite should be hidden according to the `hide`-method described in the main text of the paper. Examples of such information files can be found in the folder `input_files`. Output of this script is stored in the folder `result_files`