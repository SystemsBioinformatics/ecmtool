# ecmtool

### Installing on Linux
First make sure all dependencies are satisfied. Then execute

python3 setup.py build_ext --inplace

mv _bglu* ecmtool/


### Running with mpiexec
For example: mpiexec -n 4 python3 main.py --model_path models/e_coli_core.xml
