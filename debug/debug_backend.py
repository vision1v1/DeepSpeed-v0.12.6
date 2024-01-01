from torch import multiprocessing as mp
from torch import distributed as dist
import os

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--local-rank", type=int)

def print_mp_info():
    print(f"pid = {os.getpid()}, ppid = {os.getppid()}")
    print("cwd = ", os.getcwd())
    
def print_dist_info():
    print("initialized = ", dist.is_initialized())
    print("rank = ", dist.get_rank())
    print("world_size = ", dist.get_world_size())
    ...

def check():
    print("gloo:", dist.is_gloo_available())
    print("mpi:", dist.is_mpi_available())
    print("nccl:", dist.is_nccl_available())


def test_backend(rank, world_size):
    """
    https://pytorch.org/docs/stable/distributed.html
    https://pytorch.org/tutorials/intermediate/dist_tuto.html

    torch.distributed backend : nccl, mpi, gloo

    Use the NCCL backend for distributed GPU training
    Use the Gloo backend for distributed CPU training
    """

    print_mp_info()

    def env_init_method():
        os.environ['MASTER_ADDR'] = "localhost"
        os.environ['MASTER_PORT'] = '29500'
        return "env://"
    
    def tcp_init_method():
        return "tcp://127.0.0.1:29500"
    
    def file_init_method():
        return "file://./sharedfile"
    
    dist.init_process_group(backend="gloo", # "<device>:<backend>" 例如 "cpu:gloo"
                            init_method=env_init_method(),
                            # init_method=tcp_init_method(),
                            # init_method=file_init_method(),
                            rank=rank,
                            world_size=world_size)
    
    print_dist_info()
    
    """
    这里写训练逻辑
    """


    dist.destroy_process_group()


def test_spawn_1():
    world_size = 2
    # 产生world_size个进程
    mp.spawn(test_backend, args=(world_size,), nprocs=world_size, join=True) # rank 会自动添加
    print("finished...")


def test_spawn_2():
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=test_backend, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    check()
    test_spawn_1()
    # test_spawn_2()
    