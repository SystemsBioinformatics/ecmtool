import sys, os

if not sys.platform.startswith('win32'):
    from mpi4py import MPI
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    def get_process_rank():
        return MPI.COMM_WORLD.Get_rank()
    def is_first_process():
        return get_process_rank() == 0
else:
    # We don't have support for MPI on Windows yet, so we
    # add mock functions
    def get_process_rank():
        return 0
    def is_first_process():
        return get_process_rank() == 0