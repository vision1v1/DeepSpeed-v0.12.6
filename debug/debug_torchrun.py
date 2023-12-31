from torch import multiprocessing as mp
from torch import distributed as dist
import os


def print_mp_info():
    print(f"pid = {os.getpid()}, ppid = {os.getppid()}")
    print("cwd = ", os.getcwd())
    
def print_dist_info():
    print("initialized = ", dist.is_initialized())
    print("rank = ", dist.get_rank())
    print("world_size = ", dist.get_world_size())

def check():
    print("gloo:", dist.is_gloo_available())
    print("mpi:", dist.is_mpi_available())
    print("nccl:", dist.is_nccl_available())

def test_init():
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo")
    print("train...")

def clean():
    dist.destroy_process_group()


if __name__ == "__main__":

    # https://pytorch.org/docs/stable/elastic/run.html

    print_mp_info()
    test_init()
    print_dist_info()
    clean()

    # torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=2 debug_torchrun.py
    