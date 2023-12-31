import deepspeed
import torch

def test_init_distributed():
    """"""
    deepspeed.init_distributed(dist_backend='nccl')


def test_initialize():
    """
    初始化 DeepSpeed 引擎
    """
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(args=..., 
                                                                                      model=...,
                                                                                      model_parameters=...)



if __name__ == "__main__":
    # test_init_distributed()
    # test_initialize()
    print("finished...")
    ...