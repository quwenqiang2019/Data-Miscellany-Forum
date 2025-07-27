import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# 1. 设定随机种子
np.random.seed(42)
torch.manual_seed(42)

# 2. 读取金融时间序列数据
data=pd.read_csv("data.csv", parse_dates=["Date"], index_col=[0])
print(data)
features = data.columns.tolist()

# 3. 特征缩放与滑动窗口生成序列数据
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, target_index=0, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][target_index])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(data_scaled, seq_len=20)
split = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:split], y_seq[:split]
X_test, y_test = X_seq[split:], y_seq[split:]

# 4. 定义 LSTM 模型结构
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 5. 模型训练
model = LSTMModel(input_size=5)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

train_losses = []
for epoch in range(50):
    model.train()
    batch_losses = []
    for xb, yb in train_dl:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

# 6. 构建 KNN 模型并预测
X_train_knn = X_train.reshape(X_train.shape[0], -1)
X_test_knn = X_test.reshape(X_test.shape[0], -1)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_knn, y_train)
y_knn_pred = knn.predict(X_test_knn)

# 7. LSTM + KNN 融合预测
model.eval()
with torch.no_grad():
    y_lstm_pred = model(X_test_t).numpy().flatten()

alpha = 0.8  # 融合比例
y_fused = alpha * y_lstm_pred + (1 - alpha) * y_knn_pred

# 8. 可视化分析
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# 图1：价格与均线
plt.figure(figsize=(12, 5))
plt.plot(data['Open'], label='Open', color='blue')
plt.plot(data['High'], label='High', color='orange')
plt.plot(data['Low'], label='Low', color='green')
plt.title('Open with Moving Averages')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 图2：训练损失曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses, color='crimson')
plt.title('Training Loss Curve (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# 图3：预测对比图
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='True', color='black')
plt.plot(y_lstm_pred, label='LSTM', color='blue', linestyle='--')
plt.plot(y_knn_pred, label='KNN', color='green', linestyle=':')
plt.plot(y_fused, label='Fused', color='red')
plt.title('Prediction: True vs LSTM vs KNN vs Fused')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 图4：残差图
residuals = y_test - y_fused
plt.figure(figsize=(12, 4))
plt.bar(range(len(residuals)), residuals, color='orange')
plt.title('Prediction Residuals (True - Fused)')
plt.tight_layout()
plt.show()

# 9. 模型性能评估
mse = mean_squared_error(y_test, y_fused)
mae = mean_absolute_error(y_test, y_fused)

print(f"Fused Prediction MSE: {mse:.6f}")
print(f"Fused Prediction MAE: {mae:.6f}")