import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 1. 时间序列数据
data = pd.DataFrame(pd.read_csv('data.csv'))
print(data.head())
# 可视化原始数据
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(12, 5))
plt.plot(data['Month'], data['Passengers'], color='purple')
plt.title('Time Series Data')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.grid(True)
plt.tight_layout()
plt.show()
y = data['Passengers'].values

# 2. 构造监督学习问题（避免数据泄露）
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 7
X, y_target = create_sequences(y, seq_length)

# 分为训练集和测试集（避免未来信息泄露）
split_index = int(len(X) * 0.8)
X_train, y_train = X[:split_index], y_target[:split_index]
X_test, y_test = X[split_index:], y_target[split_index:]

# 3. 构建LSTM模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 自定义Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 初始化模型
model = LSTMRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 模型训练
num_epochs = 1000
loss_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for seqs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

# 图形1：训练损失下降曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_list, color='crimson')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 5. LSTM预测 + 决策树增强
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
lstm_preds = model(X_test_tensor).detach().numpy().flatten()

# 决策树基于LSTM残差预测
residuals = y_test - lstm_preds

# 特征扩展
X_tree_train = lstm_preds.reshape(-1, 1)
model_tree = DecisionTreeRegressor(max_depth=3)
model_tree.fit(X_tree_train, residuals)
residual_preds = model_tree.predict(X_tree_train)
final_preds = lstm_preds + residual_preds

# 图形2：LSTM与真实值对比
plt.figure(figsize=(10, 4))
plt.plot(y_test, label="True", color='blue')
plt.plot(lstm_preds, label="LSTM Prediction", color='orange')
plt.title("LSTM vs True Value")
plt.legend()
plt.grid(True)
plt.show()

# 图形3：混合模型预测对比
plt.figure(figsize=(10, 4))
plt.plot(y_test, label="True", color='blue')
plt.plot(final_preds, label="LSTM + Tree Hybrid", color='green')
plt.title("Hybrid Model vs True Value")
plt.legend()
plt.grid(True)
plt.show()

# 图形4：残差分布图（增强前后）
plt.figure(figsize=(10, 4))
sns.histplot(y_test - lstm_preds, color='red', label='LSTM Residuals', kde=True)
sns.histplot(y_test - final_preds, color='green', label='Hybrid Residuals', kde=True)
plt.title("Residual Distribution: LSTM vs Hybrid")
plt.legend()
plt.grid(True)
plt.show()

# 评估指标
print("LSTM模型 MSE:", mean_squared_error(y_test, lstm_preds))
print("混合模型 MSE:", mean_squared_error(y_test, final_preds))
print("LSTM模型 R2:", r2_score(y_test, lstm_preds))
print("混合模型 R2:", r2_score(y_test, final_preds))