import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# 1. 读取时间序列数据
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
print(df.head())

# 数据可视化: 原始时间序列
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(12, 4))
plt.plot(df['Month'], df['Passengers'], color='purple')
plt.title('原始时间序列')
plt.xlabel('时间')
plt.ylabel('数值')
plt.show()

# 3. 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df['Passengers'].values.reshape(-1,1))

# 4. 构造Transformer序列数据
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 7
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# 拆分训练集和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转为Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

# 5. 定义Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, seq_length=20):
        super(TimeSeriesTransformer, self).__init__()
        self.seq_length = seq_length
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model*seq_length, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1,0,2)  # (seq_len, batch, feature)
        x = self.transformer(x)
        x = x.permute(1,0,2).reshape(x.size(1), -1)
        out = self.fc(x)
        return out

model = TimeSeriesTransformer(seq_length=SEQ_LENGTH)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 模型训练
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.6f}')

# 7. Transformer预测
model.eval()
with torch.no_grad():
    y_pred_trans = model(X_test_t).numpy()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_trans_inv = scaler.inverse_transform(y_pred_trans)

# 8. ARIMA预测
arima_model = ARIMA(df['Passengers'][:train_size+SEQ_LENGTH], order=(5,1,0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(y_test_inv))

# 9. 融合预测
y_pred_fused = 0.8 * y_pred_trans_inv.flatten() + 0.2 * arima_forecast.values
residuals = y_test_inv.flatten() - y_pred_fused

# 10. 可视化分析
# Transformer vs 实际值
plt.figure(figsize=(14,5))
plt.plot(range(len(y_test_inv)), y_test_inv, label='真实值', color='#1f77b4', linewidth=2)
plt.plot(range(len(y_pred_trans_inv)), y_pred_trans_inv, label='Transformer预测', color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.8)
plt.scatter(range(len(y_pred_trans_inv)), y_pred_trans_inv, color='#ff7f0e', s=20, alpha=0.6)
plt.title('Transformer预测 vs 实际值')
plt.xlabel('时间步')
plt.ylabel('数值')
plt.legend()
plt.show()

# ARIMA vs 实际值
plt.figure(figsize=(14,5))
plt.plot(range(len(y_test_inv)), y_test_inv, label='真实值', color='#1f77b4', linewidth=2)
plt.plot(range(len(arima_forecast)), arima_forecast, label='ARIMA预测', color='#2ca02c', linewidth=2, linestyle='--', alpha=0.8)
plt.fill_between(range(len(arima_forecast)),
                 arima_forecast - 2*np.std(arima_forecast),
                 arima_forecast + 2*np.std(arima_forecast),
                 color='#2ca02c', alpha=0.1)
plt.title('ARIMA预测 vs 实际值')
plt.xlabel('时间步')
plt.ylabel('数值')
plt.legend()
plt.show()

# 融合模型 vs 实际值
plt.figure(figsize=(14,5))
plt.plot(range(len(y_test_inv)), y_test_inv, label='真实值', color='#1f77b4', linewidth=2)
plt.plot(range(len(y_pred_fused)), y_pred_fused, label='融合预测', color='#d62728', linewidth=2, linestyle='--')
plt.fill_between(range(len(y_pred_fused)),
                 y_pred_fused - 1.5*np.std(residuals),
                 y_pred_fused + 1.5*np.std(residuals),
                 color='#d62728', alpha=0.15)
plt.title('融合Transformer + ARIMA预测 vs 实际值')
plt.xlabel('时间步')
plt.ylabel('数值')
plt.legend()
plt.show()

# 融合模型残差分析
plt.figure(figsize=(14,5))
plt.bar(range(len(residuals)), residuals, color='#9467bd', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('融合模型残差分析')
plt.xlabel('时间步')
plt.ylabel('残差')
plt.show()