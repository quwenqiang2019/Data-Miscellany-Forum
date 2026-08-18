import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
"""多变量时序单步预测"""

# 1、读取数据集
data = pd.read_csv('/workspaces/Data-Miscellany-Forum/src/单变量时序预测系列-航空乘客预测/data.csv')
# 将日期列转换为日期时间类型
data['Month'] = pd.to_datetime(data['Month'])
# 将日期列设置为索引
data.set_index('Month', inplace=True)
series = np.array(data['Passengers'])
series = (series - series.min()) / (series.max() - series.min())
time = np.arange(len(series))

# 2. 构造滑动窗口特征
def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 7
X, y = create_windows(series, window_size)

# 3. 划分训练和测试集（避免泄露，按时间顺序切分）
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 4. 训练随机森林并提取预测值
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

# 5. 将 RF 输出拼接入 GRU 输入特征
print(X_train.shape)
X_train_gru = np.hstack([X_train, rf_train_pred.reshape(-1,1)])
print(X_train_gru.shape)
X_test_gru = np.hstack([X_test, rf_test_pred.reshape(-1,1)])

# 6. PyTorch GRU 模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# 准备 DataLoader
def to_tensor_dataset(X, y):
    return TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1),
                         torch.tensor(y, dtype=torch.float32).unsqueeze(-1))


train_dataset = to_tensor_dataset(X_train_gru, y_train)
test_dataset = to_tensor_dataset(X_test_gru, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 7. 训练 GRU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(input_size=window_size+1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    train_losses.append(epoch_loss / len(train_loader.dataset))

# 8. 测试并预测
model.eval()
with torch.no_grad():
    test_preds = []
    test_truth = []
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy().flatten()
        test_preds.extend(preds)
        test_truth.extend(y_batch.numpy().flatten())

# 9. 画图展示
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
# 图1：原始序列与训练/测试分割
plt.figure(figsize=(10,4))
plt.plot(time, series, color='magenta', label='原始序列')
print(split_idx+window_size)
plt.axvline(x=split_idx+window_size, color='cyan', linestyle='--', label='训练/测试分界')
plt.title('图1：原始时间序列与训练测试分割')
plt.legend()
plt.show()

# 图2：随机森林预测 vs 真实值（测试集）
plt.figure(figsize=(10,4))
plt.plot(range(len(y_test)), y_test, color='orange', label='真实值 y_test')
plt.plot(range(len(rf_test_pred)), rf_test_pred, color='blue', linestyle='--', label='RF 预测')
plt.title('图2：随机森林在测试集上的表现')
plt.legend()
plt.show()

# 图3：GRU 训练损失曲线
plt.figure(figsize=(10,4))
plt.plot(range(1, len(train_losses)+1), train_losses, color='red', marker='o')
plt.title('图3：GRU 训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

# 图4：混合模型最终预测 vs 真实值
plt.figure(figsize=(10,4))
plt.plot(range(len(test_truth)), test_truth, color='green', label='真实值')
plt.plot(range(len(test_preds)), test_preds, color='purple', linestyle='--', label='混合模型预测')
plt.title('图4：混合模型在测试集上的预测表现')
plt.legend()
plt.show()

# 10. 输出评价指标
mse = mean_squared_error(test_truth, test_preds)
print(f"混合模型测试集 MSE: {mse:.4f}")