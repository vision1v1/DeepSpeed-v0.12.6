import deepspeed
import torch
from torch import nn
import dataclasses
from torch.utils.data import Dataset, TensorDataset, IterableDataset, DataLoader
from torch.optim.lr_scheduler import LinearLR

def test_init_distributed():
    """"""
    deepspeed.init_distributed(dist_backend='nccl', rank=0, world_size=2)

class Args:
    local_rank = 0
    deepscale_config = {} # 参考

def test_initialize():
    """
    初始化 DeepSpeed 引擎
    参考
    https://www.deepspeed.ai/getting-started/
    """
    args = Args()
    args.local_rank = 0
    args.deepscale_config = {
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

    model = nn.Linear(in_features=6, out_features=5)
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(args=args, 
                                                                                      model=model,
                                                                                      model_parameters=model.parameters())
    
    print("training...")

class SimpleDataset(Dataset):

    def __init__(self, device, dtype) -> None:
        super().__init__()
        self.data = [torch.randn(size=(6,), device=device, dtype=dtype) for i in range(10)]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    
def ds_config_2():
    """
    参考 https://www.deepspeed.ai/docs/config-json/
    """
    optimizer_config = {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    }
    
    scheduler_config = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    }


    fp16_config = {
        "enabled": True,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "consecutive_hysteresis": False,
        "min_loss_scale": 1
    }

    return {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": optimizer_config,
        "fp16": fp16_config,
        "scheduler": scheduler_config
    }

    ...

def ds_config_1():
    """
    简单的配置
    """
    config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.00015
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": True
    }
    return config


def test_train():
    """
    训练
    """

    args = Args()
    args.local_rank = 0
    args.num_gpus = 1
    args.num_nodes = 1
    args.deepscale_config = ds_config_1()
    total_iters = 10
    # 提前在某个gpu设备上创建Tensor，后面训练可能会报错
    # 在ds_config_1 开启了float16的精度。所以这里也要设置成相同精度
    training_data = SimpleDataset(torch.device('cpu'), dtype=torch.float16) 
    model = nn.Linear(in_features=6, out_features=5)
    # optimizer = 
    # lr_scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=total_iters)
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(args=args, 
                                                                                      model=model,
                                                                                      model_parameters=model.parameters(),
                                                                                      training_data=training_data)
    
    print("model_engine ", type(model_engine))
    print("optimizer ", type(optimizer))
    print("training_dataloader ", type(training_dataloader))
    print("lr_scheduler ", type(lr_scheduler))


    print("train...")

    for step, batch in enumerate(training_dataloader):
        batch = batch.to(model_engine.device)
        #forward() method
        loss = model_engine.forward(batch) # 这里就是模型的输出，即 nn.Linear 的输出

        #runs backpropagation
        model_engine.backward(loss.sum()) # 为计算梯度转成标量

        #weight update
        model_engine.step()

    ...

def test_pytorch_train():

    model = nn.Linear(in_features=6, out_features=5)
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)

    if dataset:
        print("dataset 有数据")

    for batch in dataloader:
        output = model.forward(batch)
        print(output)
    ...

if __name__ == "__main__":
    # deepspeed --num_nodes=1 --num_gpus=1 ./debug_deepspeed.py # 需要熟悉deepspeed启动 用法
    # test_init_distributed()
    # test_initialize()
    test_train()
    # test_pytorch_train()
    print("finished...")
    ...