import sys, os

comm = None


def mpi_init():
    if sys.platform.startswith('win32'):
        print("mpi4py does not work properly on Windows, so its use will be skipped.")
        return

    try:
        from mpi4py import MPI
    except ImportError:
        print("Since the mpi4py-package is not installed, parallel computation is not possible for direct enumeration. "
              "This package does not work properly on Windows, but can be installed on other operating systems.")
        return

    global comm
    comm = MPI.COMM_WORLD
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def get_process_rank():
    return comm.Get_rank() if comm is not None else 0


def get_process_size():
    return comm.Get_size() if comm is not None else 1


def is_first_process():
    return get_process_rank() == 0


def world_allgather(data):
    return comm.allgather(data) if comm is not None else [data]


def Bcast(data, root=0):
    return comm.Bcast(data, root) if comm is not None else data


def bcast(data, root=0):
    return comm.bcast(data, root) if comm is not None else data
