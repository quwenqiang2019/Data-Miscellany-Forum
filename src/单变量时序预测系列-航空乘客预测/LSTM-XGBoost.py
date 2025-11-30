import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# 1. 数据加载
data = pd.read_csv('data.csv')
time = np.array(data['Month'])
series = np.array(data['Passengers'])

sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(12,4))
plt.plot(time, series, color='darkblue')
plt.title("虚拟时间序列数据")
plt.xlabel("时间")
plt.ylabel("值")
plt.show()

# 2. 数据预处理
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1))

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(series_scaled, seq_length)

# 拆分训练集和测试集
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转为 PyTorch 张量
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# 3. LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32)
        c0 = torch.zeros(1, x.size(0), 32)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 4. LSTM 训练
device = torch.device("cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 30  # 中低性能电脑可接受的轮数
for epoch in range(epochs):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1)%5==0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# LSTM预测
model.eval()
with torch.no_grad():
    lstm_pred = model(X_test_tensor).numpy()

# 计算残差
residuals_train = y_train - model(X_train_tensor).detach().numpy()
residuals_test = y_test - lstm_pred


# 将序列数据展开为 XGBoost特征
X_train_xgb = X_train.reshape(X_train.shape[0], -1)
X_test_xgb = X_test.reshape(X_test.shape[0], -1)

xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
xgb_model.fit(X_train_xgb, residuals_train)

xgb_pred = xgb_model.predict(X_test_xgb)
final_pred = lstm_pred + xgb_pred.reshape(-1,1)

# 5. 结果可视化
# 真实值 vs. LSTM预测
plt.figure(figsize=(14,5))
plt.plot(range(len(y_test)), scaler.inverse_transform(y_test), color='black', label='真实值')
plt.plot(range(len(y_test)), scaler.inverse_transform(lstm_pred), color='red', label='LSTM预测')
plt.plot(range(len(y_test)), scaler.inverse_transform(final_pred), color='green', label='LSTM+XGBoost预测')
plt.title("预测对比图")
plt.legend()
plt.show()
# 真实值 vs. LSTM残差
plt.figure(figsize=(12,4))
plt.plot(range(len(residuals_test)), residuals_test, color='orange')
plt.title("LSTM残差 (真实值 - LSTM预测)")
plt.xlabel("样本")
plt.ylabel("残差")
plt.show()
# LSTM残差分布
plt.figure(figsize=(12,4))
plt.hist(residuals_test, bins=30, color='purple')
plt.title("LSTM残差分布")
plt.xlabel("残差")
plt.ylabel("频数")
plt.show()
# XGBoost对LSTM残差的修正贡献
plt.figure(figsize=(12,4))
plt.plot(range(len(xgb_pred)), xgb_pred, color='cyan')
plt.title("XGBoost对LSTM残差的修正贡献")
plt.xlabel("样本")
plt.ylabel("残差修正值")
plt.show()