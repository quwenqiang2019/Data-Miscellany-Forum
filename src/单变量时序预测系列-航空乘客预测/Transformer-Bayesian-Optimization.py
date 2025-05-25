import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import optuna

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_window, output_window):
        self.series = series
        self.input_window = input_window
        self.output_window = output_window
        self.length = len(series) - input_window - output_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.input_window]
        y = self.series[idx+self.input_window:idx+self.input_window+self.output_window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_dim=1):
        super(TransformerTimeSeries, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_proj(src) * math.sqrt(self.d_model)  # 线性投影和缩放
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Transformer需要(seq_len, batch_size, d_model)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]  # 取序列最后时间步的输出
        output = self.decoder(output)  # (batch_size, output_dim)
        return output


def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.unsqueeze(-1).to(device)  # (B, seq_len, 1)
        y_batch = y_batch.squeeze(-1).to(device)  # (B)
        optimizer.zero_grad()
        output = model(x_batch).squeeze(-1)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.squeeze(-1).to(device)
            output = model(x_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            preds.append(output.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(dataloader.dataset), preds, trues


def objective(trial):
    # 超参数空间
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerTimeSeries(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 20
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, _, _ = evaluate(model, criterion, val_loader, device)
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss
    
# 设置绘图风格
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)

# 读取数据集
df = pd.DataFrame(pd.read_csv('data.csv'))
# 将日期列转换为日期时间类型
df['Month'] = pd.to_datetime(df['Month'])
df.columns = ['time','value']
df = df[['value','time']]
print(df)

# 画图：时间序列整体趋势和波动
plt.plot(df["time"], df["value"], color='blue')
plt.title("Generated Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# 划分比例
time_steps = len(df)
train_ratio = 0.8
train_size = int(time_steps * train_ratio)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# 归一化（只用训练集参数）
mean = train_df["value"].mean()
std = train_df["value"].std()

train_df["value_norm"] = (train_df["value"] - mean) / std
test_df["value_norm"] = (test_df["value"] - mean) / std

# 画图：训练集与测试集划分
plt.plot(train_df["time"], train_df["value"], label="Train", color='green')
plt.plot(test_df["time"], test_df["value"], label="Test", color='red')
plt.title("Train-Test Split of Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()



# 设置窗口大小
input_window = 2
output_window = 1

train_series = train_df["value_norm"].values
test_series = test_df["value_norm"].values

train_dataset = TimeSeriesDataset(train_series, input_window, output_window)
test_dataset = TimeSeriesDataset(test_series, input_window, output_window)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

best_params = study.best_params
final_model = TransformerTimeSeries(
    d_model=best_params["d_model"],
    nhead=best_params["nhead"],
    num_layers=best_params["num_layers"],
    dim_feedforward=best_params["dim_feedforward"],
    dropout=best_params["dropout"]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])
batch_size = best_params["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epochs = 80
train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss = train_one_epoch(final_model, optimizer, criterion, train_loader, device)
    val_loss, preds, trues = evaluate(final_model, criterion, test_loader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")


plt.plot(range(1, epochs+1), train_losses, label="Train Loss", color='blue')
plt.plot(range(1, epochs+1), val_losses, label="Validation Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

plt.plot(trues, label="True Values", color='green')
plt.plot(preds, label="Predicted Values", color='red', alpha=0.7)
plt.title("True vs Predicted Values on Test Set")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value")
plt.legend()
plt.show()

residuals = trues - preds
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("Residuals Distribution on Test Set")
plt.xlabel("Residual (True - Predicted)")
plt.ylabel("Frequency")
plt.show()


lr_values = [trial.params["lr"] for trial in study.trials]
loss_values = [trial.value for trial in study.trials]

plt.scatter(lr_values, loss_values, c=loss_values, cmap='viridis', s=80, alpha=0.8)
plt.xscale('log')
plt.colorbar(label='Validation Loss')
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Validation Loss")
plt.title("Bayesian Optimization: Learning Rate vs Validation Loss")
plt.show()





