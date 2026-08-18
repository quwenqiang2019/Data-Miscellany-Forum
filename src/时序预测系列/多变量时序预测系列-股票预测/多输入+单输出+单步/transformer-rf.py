import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

# # 读取时序数据
df = pd.DataFrame(pd.read_csv("data.csv"))
print(df)

# 可视化1：时序图（观察周期性）
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.plot(df['Date'], df['Open'], color='dodgerblue')
plt.title('Open Price Over Time')
plt.xlabel('Date')
plt.ylabel('Open')
plt.grid(True)
plt.tight_layout()
plt.show()

# 创建序列数据集
class StockDataset(Dataset):
    def __init__(self, data, input_window=24, pred_window=1):
        self.X = []
        self.y = []
        for i in range(len(data) - input_window - pred_window):
            self.X.append(data.iloc[i:i+input_window].values)
            self.y.append(data.iloc[i+input_window:i+input_window+pred_window]['Open'].values)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 选择特征
features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# 构建训练和测试数据集
input_window = 7
pred_window = 1
train_dataset = StockDataset(train_df[features], input_window, pred_window)
test_dataset = StockDataset(test_df[features], input_window, pred_window)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # 转换为(seq_len, batch, feature)
        output = self.transformer(src)
        output = output[-1]  # 最后一个时间步
        output = self.decoder(output)
        return output.squeeze()

model = TransformerModel(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练 Transformer 模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# 得到 Transformer 预测结果作为随机森林输入
model.eval()
train_features, train_targets = [], []
test_features, test_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in train_loader:
        transformer_preds = model(X_batch).numpy()
        High = X_batch[:, -1, 1].numpy()
        Low = X_batch[:, -1, 2].numpy()
        Close = X_batch[:, -1, 3].numpy()
        Adj_Close = X_batch[:, -1, 4].numpy()
        combined = np.stack([transformer_preds, High, Low, Close, Adj_Close], axis=1)
        train_features.extend(combined)
        train_targets.extend(y_batch.numpy())

    for X_batch, y_batch in test_loader:
        transformer_preds = model(X_batch).numpy()
        High = X_batch[:, -1, 1].numpy()
        Low = X_batch[:, -1, 2].numpy()
        Close = X_batch[:, -1, 3].numpy()
        Adj_Close = X_batch[:, -1, 4].numpy()
        combined = np.stack([transformer_preds, High, Low, Close, Adj_Close], axis=1)
        test_features.extend(combined)
        test_targets.extend(y_batch.numpy())

# 随机森林模型训练
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_features, np.array(train_targets).ravel())
rf_preds = rf.predict(test_features)

# 可视化2：预测 vs 实际
plt.figure(figsize=(14, 4))
plt.plot(np.array(test_targets).ravel(), label='Actual', color='black')
plt.plot(rf_preds, label='Predicted', color='crimson')
plt.title('Actual vs Predicted Open Price')
plt.xlabel('Time Index')
plt.ylabel('Open Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 可视化3：误差分布图
errors = rf_preds - np.array(test_targets).ravel()
sns.histplot(errors, bins=30, kde=True, color='darkorange')
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 可视化4：特征重要性
feature_importances = rf.feature_importances_
print(feature_importances)
feature_names = ['Transformer Output', 'High', 'Low', 'Close', 'Adj_Close']
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# 模型评估指标
mse = mean_squared_error(test_targets, rf_preds)
print(f"Random Forest Ensemble MSE: {mse:.2f}")