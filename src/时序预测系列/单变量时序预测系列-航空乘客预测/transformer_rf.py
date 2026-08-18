import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns

# 1. 时间序列数据
data = pd.read_csv('data.csv')
data['Month'] = pd.to_datetime(data['Month'])# 将日期列转换为日期时间类型
t = np.array(data['Month'])
x = np.array(data['Passengers'])
T= len(data['Month'])

sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(10,4))
plt.plot(t, x, color='purple')
plt.title("图1：原始时间序列")
plt.xlabel("时间步")
plt.ylabel("值")
plt.show()

# 2. 构建时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=20):
        self.series = series
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.series) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

seq_len = 20
dataset = TimeSeriesDataset(x, seq_len)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 构建 Transformer 特征提取器
class TimeSeriesTransformer(nn.Module):
    def __init__(self, seq_len, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化得到固定长度特征
    
    def forward(self, x):
        x = x.unsqueeze(-1)        # [B, L, 1]
        x = self.embedding(x)      # [B, L, d_model]
        x = self.transformer(x)    # [B, L, d_model]
        x = x.transpose(1,2)       # [B, d_model, L]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        return x

model = TimeSeriesTransformer(seq_len)

# 4. 提取训练/验证特征
def extract_features(model, loader):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            h = model(x_batch)
            features.append(h.numpy())
            targets.append(y_batch.numpy())
    return np.vstack(features), np.hstack(targets)

train_features, train_targets = extract_features(model, train_loader)
val_features, val_targets = extract_features(model, val_loader)

# 5. 随机森林训练与预测
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_features, train_targets)
pred = rf.predict(val_features)

mse = mean_squared_error(val_targets, pred)
r2 = r2_score(val_targets, pred)
print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

# 6. 可视化分析

# 图2：预测 vs 真实
plt.figure(figsize=(12,5))
plt.plot(val_targets, label='真实值', color='orange')
plt.plot(pred, label='预测值', color='cyan')
plt.title("图2：Transformer + 随机森林预测 vs 真实值")
plt.xlabel("验证集时间步")
plt.ylabel("值")
plt.legend()
plt.show()

# 图3：预测误差随时间
plt.figure(figsize=(12,4))
plt.plot(val_targets - pred, color='red')
plt.title("图3：预测误差随时间变化")
plt.xlabel("时间步")
plt.ylabel("误差")
plt.show()

# 图4：随机森林特征重要性
importances = rf.feature_importances_
plt.figure(figsize=(10,4))
plt.bar(range(len(importances)), importances, color='green')
plt.title("图4：随机森林特征重要性（Transformer输出特征）")
plt.xlabel("特征索引")
plt.ylabel("重要性")
plt.show()

# 图5：残差分布直方图
plt.figure(figsize=(8,4))
plt.hist(val_targets - pred, bins=30, color='magenta')
plt.title("图5：残差分布直方图")
plt.xlabel("残差")
plt.ylabel("频数")
plt.show()