import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 时间序列数据
# data = pd.read_csv('data.csv')
data = pd.read_csv('/workspaces/Data-Miscellany-Forum/src/单变量时序预测系列-航空乘客预测/data.csv')
print(data)
t = np.array(data['Month'])
series = np.array(data['Passengers'])
series = series.astype(np.float32)

# 2. 构建 Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_len=30, pred_len=5):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len
        self.len = len(data) - input_len - pred_len + 1
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.pred_len]
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)

input_len = 30
pred_len = 5
dataset = TimeSeriesDataset(series, input_len, pred_len)
print(dataset.__getitem__)
# 拆分训练/测试集
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 时间序列通常不要shuffle
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 3. 定义融合模型：CNN + LSTM + Transformer
class CNN_LSTM_Transformer(nn.Module):
    def __init__(self, input_dim=1, cnn_channels=16, lstm_hidden=32, transformer_dim=32,
                 transformer_heads=4, transformer_layers=1, pred_len=5):
        super().__init__()
        # CNN
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.cnn_relu = nn.ReLU()
        
        # LSTM
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, batch_first=True)
        
        # Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Projection layers
        self.proj_lstm = nn.Linear(lstm_hidden, transformer_dim)
        self.pred_len = pred_len
        self.fc_out = nn.Linear(transformer_dim, pred_len)
    
    def forward(self, x):
        # x: [batch, seq_len, 1]
        batch_size, seq_len, _ = x.shape
        # CNN expects [batch, channels, seq_len]
        cnn_out = self.cnn_relu(self.cnn(x.transpose(1,2)))  # [B, C, T]
        cnn_out = cnn_out.transpose(1,2)  # [B, T, C]
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # [B, T, hidden]
        lstm_proj = self.proj_lstm(lstm_out)  # [B, T, transformer_dim]
        # Transformer
        trans_out = self.transformer(lstm_proj)  # [B, T, transformer_dim]
        # 取最后时间步输出预测
        out = self.fc_out(trans_out[:, -1, :])  # [B, pred_len]
        return out.unsqueeze(-1)  # [B, pred_len, 1]

# 4. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/train_size:.4f}")

# 5. 测试集预测
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy())
        trues.append(yb.numpy())
preds = np.concatenate(preds, axis=0).squeeze(-1)  # [num_samples, pred_len]
trues = np.concatenate(trues, axis=0).squeeze(-1)

# 6. 可视化分析
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# 图1：原始时间序列
time_axis = np.arange(len(series))
plt.figure()
plt.plot(time_axis, series, color='royalblue')
plt.title("图1：原始时间序列")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# 图2：训练/测试集分布
plt.figure()
plt.plot(time_axis[:len(series[:train_size+input_len+pred_len])], series[:train_size+input_len+pred_len], color='green', label='Train')
plt.plot(time_axis[len(series[:train_size+input_len+pred_len]):], series[len(series[:train_size+input_len+pred_len]):], color='red', label='Test')
plt.title("图2：训练集与测试集时间序列分布")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# 图3：预测对比（测试集前50个样本）
plt.figure()
plt.plot(trues[:50].flatten(), color='black', label='True')
plt.plot(preds[:50].flatten(), color='orange', linestyle='--', label='Predicted')
plt.title("图3：测试集预测对比（前50步）")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.show()

# 图4：预测残差分布
residuals = trues - preds
plt.figure()
sns.histplot(residuals.flatten(), bins=30, kde=True, color='purple')
plt.title("图4：预测残差分布（True - Pred）")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()