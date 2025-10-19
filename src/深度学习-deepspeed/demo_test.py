import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# DeepSpeed配置
deepspeed_config = {
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "steps_per_print": 200,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 50000000,
        "overlap_comm": True
    },
    "fp16": {
        "enabled": True,
        "loss_scale": "dynamic"
    }
}
# 初始化模型、优化器和DeepSpeed
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=None,
    model=model,
    optimizer=optimizer,
    config=deepspeed_config
)
# 训练过程示例
def train_epoch(data_loader):
    model_engine.train()
    for batch in data_loader:
        inputs, labels = batch
        model_engine.zero_grad()
        outputs = model_engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()
# 假设你已经有一个数据加载器data_loader
# train_epoch(data_loader)