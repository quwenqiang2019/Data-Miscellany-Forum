import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
# 读取数据集
df = pd.DataFrame(pd.read_csv('data.csv'))
df['Month'] = pd.to_datetime(df['Month'])
print(df.head())
# 数据可视化
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(10, 5))
plt.plot(df['Month'], df['Passengers'], label='Time Series')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.title('Time Series Data')
plt.show()

# 数据预处理
df['lag1'] = df['Passengers'].shift(1)
df['lag2'] = df['Passengers'].shift(2)
df = df.dropna()
# 数据准备
X = df[['lag1', 'lag2']].values
y = df['Passengers'].values
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
# 数据加载
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型定义
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out

model = LSTMAttention(input_dim=2, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 10000
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(1)  # 添加时间步维度
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')


model.eval()
X_test = X.unsqueeze(1)
y_pred = model(X_test).detach().numpy()

plt.figure(figsize=(12,6))
plt.plot(df['Month'], df['Passengers'], label='True Values', color='blue')
plt.plot(df['Month'], y_pred, label='Predicted Values', color='red', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.title('Model Prediction vs. True Values')
plt.show()