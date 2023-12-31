from torch import distributed as dist
import os
print("pid = ", os.getpid())
print("cwd = ", os.getcwd())

def check():
    print("gloo:", dist.is_gloo_available())
    print("mpi:", dist.is_mpi_available())
    print("nccl:", dist.is_nccl_available())


def test_backend():
    """
    https://pytorch.org/docs/stable/distributed.html
    torch.distributed backend : nccl, mpi, gloo

    Use the NCCL backend for distributed GPU training
    Use the Gloo backend for distributed CPU training
    """

    def env_init_method():
        os.environ['MASTER_ADDR'] = "localhost"
        os.environ['MASTER_PORT'] = '12355'
        return "env://"
    
    def tcp_init_method():
        return "tcp://127.0.0.1:12355"
    
    def file_init_method():
        return "file://./sharedfile"
    
    dist.init_process_group(backend="gloo", # "<device>:<backend>" 例如 "cpu:gloo"
                            # init_method=env_init_method(),
                            # init_method=tcp_init_method(),
                            init_method=file_init_method(),
                            rank=0,
                            world_size=2)
    ...


if __name__ == "__main__":
    check()
    test_backend()
    print("finished...")