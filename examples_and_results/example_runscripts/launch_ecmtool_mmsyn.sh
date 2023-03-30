#!/bin/bash

#SBATCH --job-name=ecmtool         # create a short name for your job
# #SBATCH --nodes=2                # node count
#SBATCH --ntasks=20               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end,FAIL          # send email when job ends
#SBATCH --mail-user=my.email@unibas.ch
#SBATCH --qos=6hours
#SBATCH --output=./ecmtoolSingularity.out     #These are the STDOUT and STDERR files
#SBATCH --error=./ecmtoolSingularity.err


# Arguments for ecmtool run are defined here
MODEL_PATH="models/mmsyn_xml_files/mmsyn_sm10.xml"
OUT_PATH="tmp/conversion_cone_mmsyn_sm10.csv"
ADD_OBJ="true"
AUTO_DIRECT="true"
EXT_COMP="None"
HIDE=""
PROHIBIT=""
TAG=""
INPUTS=""
OUTPUTS=""
COMPRESS="true"
SPLIT_BEF_DD="true"
HIDE_ALL_IN_OR_OUTPUTS=""

# Some recurring commands are defined here
SINGULARITY_CMD="singularity exec --pwd /ecmtool --bind results:/ecmtool/tmp,results:/ecmtool/ecmtool/tmp,results:/scratch ecmtool_latest.sif"
PY_COMMAND="srun --ntasks=1 --nodes=1 ${SINGULARITY_CMD} python3 main.py"
MPLRS_COMMAND1="mpirun -np ${SLURM_NTASKS} ${SINGULARITY_CMD} mplrs -redund tmp/mplrs.ine tmp/redund.ine"
MPLRS_COMMAND2="mpirun -np ${SLURM_NTASKS} ${SINGULARITY_CMD}  mplrs tmp/redund.ine tmp/mplrs.out"
PARALLEL_PY_COMMAND="mpirun -np ${SLURM_NTASKS} ${SINGULARITY_CMD} python3 main.py"

module purge
module load OpenMPI/4.1.1-GCC-10.3.0
export PMIX_MCA_gds=hash
${PY_COMMAND} preprocess --model_path ${MODEL_PATH} --auto_direction ${AUTO_DIRECT} --hide "${HIDE}" --prohibit "${PROHIBIT}" --tag "${TAG}" --inputs "${INPUTS}" --outputs "${OUTPUTS}" --use_external_compartment "${EXT_COMP}" --add_objective_metabolite "${ADD_OBJ}" --compress "${COMPRESS}" --splitting_before_polco ${SPLIT_BEF_DD} --hide_all_in_or_outputs "${HIDE_ALL_IN_OR_OUTPUTS}" 
${PY_COMMAND} calc_linearities
${PY_COMMAND} prep_C0_rays
${MPLRS_COMMAND1}
${MPLRS_COMMAND2}
${PY_COMMAND} process_C0_rays
${PARALLEL_PY_COMMAND} calc_H
${PY_COMMAND} prep_C_rays
${MPLRS_COMMAND1}
${MPLRS_COMMAND2}
${PY_COMMAND} process_C_rays
${PY_COMMAND} postprocess
${PY_COMMAND} save_ecms --out_path ${OUT_PATH}
