import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import deepspeed

# 生成模拟的心脏疾病数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                           n_classes=2, weights=[0.5, 0.5], random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据转换为TensorDataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义一个简单的二分类神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid激活函数进行二分类
        return x


# 检查是否有可用的GPU，如果有则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、优化器
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

# 使用DeepSpeed初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=None,
    model=model,
    optimizer=optimizer,
    config=deepspeed_config
)


# 训练过程示例
def train_epoch(data_loader):
    model_engine.train()
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        # 将数据移动到相应的设备
        inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)  # 将标签转换为浮点数并调整形状

        # 梯度清零
        model_engine.zero_grad()
        # 前向传播
        outputs = model_engine(inputs)
        # 计算损失
        loss = nn.BCELoss()(outputs, labels)
        # 反向传播
        model_engine.backward(loss)
        # 更新参数
        model_engine.step()
        # 打印损失
        if batch_idx % deepspeed_config["steps_per_print"] == 0:
            print(f"Step {batch_idx}, Loss: {loss.item()}")


# 测试过程示例
def test_epoch(data_loader):
    model_engine.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # 将数据移动到相应的设备
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)  # 将标签转换为浮点数并调整形状

            # 前向传播
            outputs = model_engine(inputs)
            # 计算损失
            test_loss += nn.BCELoss()(outputs, labels).item()
            # 计算准确率
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 平均损失和准确率
    test_loss /= len(data_loader.dataset)
    correct /= len(data_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct:.4f}")


# 训练和测试模型
for epoch in range(1, 6):  # 训练5个周期
    print(f"Epoch {epoch}")
    train_epoch(train_loader)
    test_epoch(test_loader)
