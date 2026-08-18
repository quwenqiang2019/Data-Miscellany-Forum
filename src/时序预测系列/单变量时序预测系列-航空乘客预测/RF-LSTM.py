import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 设置随机种子
SEED = 42

# 第一部分：读取时间序列数据
def read_data(filename='data.csv'):
    # 1. 数据读取部分
    df = pd.DataFrame(pd.read_csv(filename))
    return df

data = read_data()
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

# 第二部分：特征工程（用于随机森林）
def create_features(df, lags=7):
    df_feat = df.copy()
    for lag in range(1, lags + 1):
        df_feat[f'lag_{lag}'] = df_feat['Passengers'].shift(lag)
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

df_rf = create_features(data, lags=7)


# 划分训练集与测试集
train_size = int(len(df_rf) * 0.8)
train_rf = df_rf.iloc[:train_size]
test_rf = df_rf.iloc[train_size:]

# 随机森林训练与预测
rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
X_train = train_rf.drop(['Passengers', 'Month'], axis=1)
y_train = train_rf['Passengers']
X_test = test_rf.drop(['Passengers', 'Month'], axis=1)
y_test = test_rf['Passengers']

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 第三部分：LSTM建模
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_length):
        self.series = series
        self.seq_length = seq_length

    def __len__(self):
        return len(self.series) - self.seq_length

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.seq_length]
        y = self.series[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# 归一化处理
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(data[['Passengers']].values).flatten()
seq_length = 14

# 分割数据
train_seq = scaled_values[:int(len(scaled_values) * 0.8)]
test_seq = scaled_values[int(len(scaled_values) * 0.8) - seq_length:]

train_dataset = TimeSeriesDataset(train_seq, seq_length)
test_dataset = TimeSeriesDataset(test_seq, seq_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 训练模型
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# LSTM预测
model.eval()
pred_lstm = []
true_lstm = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.unsqueeze(-1)
        output = model(x_batch)
        pred_lstm.append(output.item())
        true_lstm.append(y_batch.item())

pred_lstm_rescaled = scaler.inverse_transform(np.array(pred_lstm).reshape(-1, 1)).flatten()
true_lstm_rescaled = scaler.inverse_transform(np.array(true_lstm).reshape(-1, 1)).flatten()

# 第四部分：融合模型
min_len = min(len(pred_lstm_rescaled), len(y_pred_rf), len(y_test))

pred_lstm_rescaled = pred_lstm_rescaled[:min_len]
y_pred_rf = y_pred_rf[:min_len]
true_vals = y_test.values[:min_len]

fusion_pred = (pred_lstm_rescaled + y_pred_rf) / 2

# 第五部分：结果分析与可视化
plt.figure(figsize=(12, 5))
plt.plot(true_vals, label='True Values', color='black')
plt.plot(y_pred_rf, label='Random Forest', color='green')
plt.plot(pred_lstm_rescaled, label='LSTM', color='blue')
plt.plot(fusion_pred, label='Fusion Model', color='red', linestyle='--')
plt.title('Model Comparison on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 均方误差比较
mse_rf = mean_squared_error(true_vals, y_pred_rf)
mse_lstm = mean_squared_error(true_vals, pred_lstm_rescaled)
mse_fusion = mean_squared_error(true_vals, fusion_pred)

# 图2：误差柱状图
plt.figure(figsize=(8, 4))
plt.bar(['Random Forest', 'LSTM', 'Fusion'], [mse_rf, mse_lstm, mse_fusion], color=['green', 'blue', 'red'])
plt.title('Mean Squared Error Comparison')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()

# 图3：误差分布图
error_df = pd.DataFrame({
    'Random Forest': y_pred_rf - true_vals,
    'LSTM': pred_lstm_rescaled - true_vals,
    'Fusion': fusion_pred - true_vals
})
plt.figure(figsize=(10, 5))
sns.histplot(error_df, kde=True, palette='bright')
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.tight_layout()
plt.show()