import deepspeed
import torch

def test_init_distributed():
    """"""
    deepspeed.init_distributed(dist_backend='nccl', rank=0, world_size=2)


def test_initialize():
    """
    初始化 DeepSpeed 引擎
    参考
    https://www.deepspeed.ai/getting-started/
    """
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(args=..., 
                                                                                      model=...,
                                                                                      model_parameters=...)
    
    print("training...")



if __name__ == "__main__":
    # test_init_distributed()
    test_initialize()
    print("finished...")
    ...