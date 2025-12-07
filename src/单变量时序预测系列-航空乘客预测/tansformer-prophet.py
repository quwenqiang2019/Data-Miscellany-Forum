import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 数据集生成
np.random.seed(42)
days = 365
t = np.arange(days)

trend = 0.05 * t
seasonality = 10 * np.sin(2 * np.pi * t / 30)  # 月度周期
noise = np.random.normal(0, 2, days)
sales = 50 + trend + seasonality + noise

df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=days),
    'y': sales
})

plt.figure(figsize=(12,5))
plt.plot(df['ds'], df['y'], label='原始销售量', color='dodgerblue')
plt.plot(df['ds'], 50 + trend, label='线性趋势', color='orange')
plt.title("图1: 原始销售量与线性趋势")
plt.xlabel("日期")
plt.ylabel("销售量")
plt.legend()
plt.show()


# 2. Prophet预测
train_df = df.iloc[:-30]
test_df = df.iloc[-30:]

prophet_model = Prophet(yearly_seasonality=False,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_mode='additive')
prophet_model.add_seasonality(name='monthly', period=30, fourier_order=5)
prophet_model.fit(train_df)

future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

plt.figure(figsize=(12,5))
plt.plot(df['ds'], df['y'], label='原始销售量', color='dodgerblue')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet预测', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
plt.title("图2: Prophet拟合及预测")
plt.xlabel("日期")
plt.ylabel("销售量")
plt.legend()
plt.show()

# 3. Transformer预测残差
residual = train_df['y'].values - forecast['yhat'].iloc[:len(train_df)].values
mean_r, std_r = residual.mean(), residual.std()
residual_norm = (residual - mean_r) / std_r  # 标准化

seq_len = 30
X, y = [], []
for i in range(len(residual_norm) - seq_len):
    X.append(residual_norm[i:i+seq_len])
    y.append(residual_norm[i+seq_len])

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class ResidualTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x.permute(1,0,2))
        out = self.fc_out(x[-1])
        return out

device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
model = ResidualTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 4. 训练Transformer
epochs = 50
model.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 5. Transformer预测未来残差
model.eval()
preds_residual = []
input_seq = residual_norm[-seq_len:]
input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

for _ in range(30):
    with torch.no_grad():
        pred = model(input_seq)
        preds_residual.append(pred.item())
        pred_step = pred.unsqueeze(1)
        input_seq = torch.cat([input_seq[:,1:,:], pred_step], dim=1)

preds_residual = np.array(preds_residual) * std_r + mean_r

# 6. 融合预测
fusion_pred = forecast['yhat'].iloc[-30:].values + preds_residual

plt.figure(figsize=(12,5))
plt.plot(test_df['ds'], residual[-30:], label='实际残差', color='orange')
plt.plot(test_df['ds'], preds_residual, label='Transformer残差预测', color='green')
plt.title("图3: Transformer残差预测")
plt.xlabel("日期")
plt.ylabel("残差")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df['ds'], df['y'], label='原始销售量', color='dodgerblue')
plt.plot(test_df['ds'], fusion_pred, label='融合预测', color='purple', linewidth=2)
plt.title("图4: 融合 Prophet + Transformer预测对比实际值")
plt.xlabel("日期")
plt.ylabel("销售量")
plt.legend()
plt.show()