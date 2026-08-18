import numpy as np
import pywt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns


# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

# Step 1: 数据读取
data = pd.read_csv('data.csv')
# 将日期列转换为日期时间类型
data['Month'] = pd.to_datetime(data['Month'])
# 将日期列设置为索引
data.set_index('Month', inplace=True)
data = np.array(data['Passengers'])
T = len(data)
x = np.arange(len(data))


# 可视化：原始时间序列
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(10, 4))
plt.plot(x, data, color='darkorange')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 2: 小波分解 + 数据标准化
wavelet = 'db4'
coeffs = pywt.wavedec(data, wavelet, level=2)
a2, d2, d1 = coeffs

# 重构原始信号
low_freq  = pywt.waverec([a2, np.zeros_like(d2), np.zeros_like(d1)], wavelet)
mid_freq  = pywt.waverec([np.zeros_like(a2), d2, np.zeros_like(d1)], wavelet)
high_freq = pywt.waverec([np.zeros_like(a2), np.zeros_like(d2), d1], wavelet)
print(low_freq.shape, mid_freq.shape, high_freq.shape)

# 统一长度
min_len = min(len(low_freq), T)
components = np.stack([low_freq[:min_len], mid_freq[:min_len], high_freq[:min_len]], axis=1)

# 标准化
scaler = MinMaxScaler()
components_scaled = scaler.fit_transform(components)
target = data[1:1+min_len]  # 滞后2步预测，避免数据泄露

# Step 3: 构造 PyTorch 数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = []
        self.y = []
        for i in range(len(X) - window_size):
            self.X.append(X[i:i+window_size])
            self.y.append(y[i+window_size])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

window_size = 7
offset = 1
X_all = components_scaled[:-offset]            # shape: (min_len-, 3)
y_all = data[offset : offset + len(X_all)]     # shape: (min_len-,)
dataset = TimeSeriesDataset(X_all, y_all, window_size)

train_size = int(len(dataset) * 0.8)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Step 4: 构建 Transformer 模型
class WaveletTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.linear_in(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.decoder(x).squeeze(-1)
        return out

model = WaveletTransformer(input_dim=3)

# Step 5: 模型训练
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        test_loss = sum(criterion(model(X), y).item() for X, y in test_loader) / len(test_loader)
        test_losses.append(test_loss)

# 可视化：训练曲线
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss', color='royalblue')
plt.plot(test_losses, label='Test Loss', color='tomato')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: 模型预测与结果分析
model.eval()
preds = []
y_true = []
with torch.no_grad():
    for X, y in test_loader:
        pred = model(X)
        preds.extend(pred.numpy())
        y_true.extend(y.numpy())

# 可视化：预测 vs 真实
plt.figure(figsize=(10, 4))
plt.plot(y_true, label='True', color='green')
plt.plot(preds, label='Predicted', color='purple')
plt.title('Prediction vs Ground Truth')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# 可视化：不同频段成分
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(components[:min_len, 0], color='blue')
plt.title('Low Frequency Component')
plt.subplot(3, 1, 2)
plt.plot(components[:min_len, 1], color='orange')
plt.title('Mid Frequency Component')
plt.subplot(3, 1, 3)
plt.plot(components[:min_len, 2], color='red')
plt.title('High Frequency Component')
plt.tight_layout()
plt.show()

# 可视化：预测误差分布
errors = np.array(preds) - np.array(y_true)
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=40, color='steelblue', edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()