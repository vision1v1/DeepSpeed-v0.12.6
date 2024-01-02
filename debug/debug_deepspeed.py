import deepspeed
import torch
from torch import nn
import dataclasses

def test_init_distributed():
    """"""
    deepspeed.init_distributed(dist_backend='nccl', rank=0, world_size=2)

class Args:
    local_rank = 0
    deepscale_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": True
    }

def test_initialize():
    """
    初始化 DeepSpeed 引擎
    参考
    https://www.deepspeed.ai/getting-started/
    """
    args = Args()
    model = nn.Linear(in_features=6, out_features=5)
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(args=args, 
                                                                                      model=model,
                                                                                      model_parameters=model.parameters())
    
    print("training...")



if __name__ == "__main__":
    # test_init_distributed()
    test_initialize()
    print("finished...")
    ...