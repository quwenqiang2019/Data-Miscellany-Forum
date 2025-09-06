import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 读取金融时间序列数据
df = pd.DataFrame(pd.read_csv("data.csv"))
df["Open"] = df.pop("Open")
print(df)

# 2. 数据预处理
features = ['High', 'Low', 'Close', 'Adj Close']
target = 'Open'
scaler = MinMaxScaler()
df[['High', 'Low', 'Close', 'Adj Close', 'Open']] = scaler.fit_transform(df[['High', 'Low', 'Close', 'Adj Close', 'Open']])

# 3. 构造时间序列数据集
seq_length = 10
X, y = [], []
for i in range(len(df) - seq_length):
    X.append(df[features].iloc[i:i+seq_length].values)
    y.append(df[target].iloc[i+seq_length])
X, y = np.array(X), np.array(y)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 6. 训练 LSTM
input_dim = 4
hidden_dim = 64
output_dim = 1
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

# 训练循环
epochs = 100
train_losses = []
for epoch in range(epochs):
    lstm_model.train()
    optimizer.zero_grad()
    output = lstm_model(X_train_torch)
    loss = criterion(output.squeeze(), y_train_torch)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

# 7. LSTM 特征提取
lstm_model.eval()
lstm_features = lstm_model(X_train_torch).detach().cpu().numpy()
lstm_features_test = lstm_model(X_test_torch).detach().cpu().numpy()

# 8. LightGBM 训练
train_features = np.hstack((X_train.reshape(X_train.shape[0], -1), lstm_features))
test_features = np.hstack((X_test.reshape(X_test.shape[0], -1), lstm_features_test))

lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
lgb_model.fit(train_features, y_train)

y_pred = lgb_model.predict(test_features)


# 9. 评估
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# (1) 时间序列趋势
# plt.plot(df['Date'], df[target], label='Target', color='blue')
plt.plot(range(len(df[target])), df[target], color='blue')
plt.title('Time Series Trend')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# (2) LSTM 训练损失
plt.plot(range(epochs), train_losses, color='red')
plt.title('LSTM Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# (3) LightGBM 特征重要性
lgb.plot_importance(lgb_model, importance_type='gain', color='green')
plt.title('LightGBM Feature Importance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# (4) 预测结果 vs 真实值
plt.plot(y_test, label='Actual Value', color='black')
plt.plot(y_pred, label='Predicted Value', linestyle='dashed', color='orange')
plt.legend()
plt.title('Prediction vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
