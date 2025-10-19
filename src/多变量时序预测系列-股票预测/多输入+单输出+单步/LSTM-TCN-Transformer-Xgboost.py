import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


data_ = pd.read_csv("data.csv")
df = pd.DataFrame(data_)
print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
print(type(df['Date']))
features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
data = df[features].values
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# # 1. 时间序列数据
# np.random.seed(42)
# n = 1200
# t = np.arange(n)
# trend = 0.05 * t
# seasonal = 10 * np.sin(2 * np.pi * t / 50)
# noise = np.random.normal(0, 2, n)
# event = np.zeros(n)
# event[300:310] += 20
# event[800:810] -= 15
# series = trend + seasonal + noise + event
#
# df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=n), "y": series})

# plt.figure(figsize=(14,5))
# plt.plot(df['Month'], df['Passengers'], color='#FF6347', label='原始时间序列')
# plt.title('原始虚拟时间序列数据')
# plt.xlabel('时间')
# plt.ylabel('数值')
# plt.legend()
# plt.show()
#
# # 2. 数据增强
# # 2.1 滞后特征和滚动统计
# window = 60
# df['y_roll_mean'] = df['y'].rolling(5).mean().fillna(method='bfill')
# df['y_roll_std'] = df['y'].rolling(5).std().fillna(method='bfill')
#
# # 2.2 周期性特征
# df['sin_50'] = np.sin(2 * np.pi * t / 50)
# df['cos_50'] = np.cos(2 * np.pi * t / 50)
# df['sin_200'] = np.sin(2 * np.pi * t / 200)
# df['cos_200'] = np.cos(2 * np.pi * t / 200)
#
# # 2.3 事件特征
# df['event'] = 0
# df.loc[300:310, 'event'] = 1
# df.loc[800:810, 'event'] = -1
#
# features = ['y', 'y_roll_mean', 'y_roll_std', 'sin_50', 'cos_50', 'sin_200', 'cos_200', 'event']
# data = df[features].values
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)





# 3. 构建序列
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len,0])  # 预测原始 y
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(data_scaled, seq_len)

train_size = int(0.7*len(X))
val_size = int(0.2*len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 4. 构建增强版模型
# 4.1 LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out

# 4.2 Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size,d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.fc = nn.Linear(d_model,1)
    def forward(self,x):
        x = self.input_proj(x)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x[-1,:,:]
        return self.fc(x)

# 4.3 TCN
class TemporalBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,dilation,dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,
                                           padding=(kernel_size-1)*dilation,dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels,out_channels,kernel_size,stride=stride,
                                           padding=(kernel_size-1)*dilation,dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels,out_channels,1) if in_channels!=out_channels else None
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        if out.size(2) > x.size(2):
            out = out[:,:,-x.size(2):]
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) > out.size(2):
            res = res[:,:,-out.size(2):]
        return self.relu(out+res)

class TCN(nn.Module):
    def __init__(self,num_inputs,num_channels=[64,64],kernel_size=3):
        super().__init__()
        layers=[]
        num_levels=len(num_channels)
        for i in range(num_levels):
            dilation_size=2**i
            in_ch=num_inputs if i==0 else num_channels[i-1]
            out_ch=num_channels[i]
            layers.append(TemporalBlock(in_ch,out_ch,kernel_size,stride=1,dilation=dilation_size,dropout=0.2))
        self.network=nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1],1)
    def forward(self,x):
        x = x.permute(0,2,1)
        y = self.network(x)
        y = y[:,:,-1]
        return self.fc(y)

input_size = X_train.shape[2]
lstm_model = LSTMModel(input_size)
transformer_model = TransformerModel(input_size)
tcn_model = TCN(input_size)

# 5. 训练函数
def train_model(model,X_train,y_train,X_val,y_val,epochs=30,lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred,y_val)
        if (epoch+1)%10==0:
            print(f'Epoch {epoch+1}, Train Loss:{loss.item():.4f}, Val Loss:{val_loss.item():.4f}')
    return model

lstm_model = train_model(lstm_model,X_train_t,y_train_t,X_val_t,y_val_t)
transformer_model = train_model(transformer_model,X_train_t,y_train_t,X_val_t,y_val_t)
tcn_model = train_model(tcn_model,X_train_t,y_train_t,X_val_t,y_val_t)

# 6. 基模型预测
lstm_pred = lstm_model(X_test_t).detach().numpy()
transformer_pred = transformer_model(X_test_t).detach().numpy()
tcn_pred = tcn_model(X_test_t).detach().numpy()

y_test_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), np.zeros((len(y_test),4))]))[:,0]
lstm_pred_inv = scaler.inverse_transform(np.hstack([lstm_pred,np.zeros((len(lstm_pred),4))]))[:,0]
transformer_pred_inv = scaler.inverse_transform(np.hstack([transformer_pred,np.zeros((len(transformer_pred),4))]))[:,0]
tcn_pred_inv = scaler.inverse_transform(np.hstack([tcn_pred,np.zeros((len(tcn_pred),4))]))[:,0]

plt.figure(figsize=(14,5))
plt.plot(df['Date'][-len(y_test):],y_test_inv,label='真实值',color='#FF6347')
plt.plot(df['Date'][-len(y_test):],lstm_pred_inv,label='LSTM预测',color='#1E90FF')
plt.plot(df['Date'][-len(y_test):],transformer_pred_inv,label='Transformer预测',color='#32CD32')
plt.plot(df['Date'][-len(y_test):],tcn_pred_inv,label='TCN预测',color='#FFD700')
plt.title('基模型预测对比')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.show()

# 7. 残差 stacking 融合
stack_val_X = np.hstack([lstm_model(X_val_t).detach().numpy(),
                         transformer_model(X_val_t).detach().numpy(),
                         tcn_model(X_val_t).detach().numpy()])
stack_val_res = y_val_t.detach().numpy() - stack_val_X.mean(axis=1,keepdims=True)

stack_test_X = np.hstack([lstm_pred,transformer_pred,tcn_pred])

xgb_model = xgb.XGBRegressor(n_estimators=200,learning_rate=0.1)
xgb_model.fit(stack_val_X, stack_val_res)
stack_res_pred = xgb_model.predict(stack_test_X).reshape(-1,1)

# 加权融合
stack_pred_final = (lstm_pred + transformer_pred + tcn_pred)/3 + stack_res_pred

stack_pred_inv = scaler.inverse_transform(np.hstack([stack_pred_final,np.zeros((len(stack_pred_final),4))]))[:,0]

plt.figure(figsize=(14,5))
plt.plot(df['Date'][-len(y_test):],y_test_inv,label='真实值',color='#FF6347')
plt.plot(df['Date'][-len(y_test):],stack_pred_inv,label='融合预测',color='#8A2BE2')
plt.title('残差 stacking 融合预测')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.show()

# 8. 残差分布分析
residual = y_test_inv - stack_pred_inv
plt.figure(figsize=(12,5))
plt.hist(residual,bins=50,color='#FF4500',alpha=0.7)
plt.title('融合模型残差分布')
plt.xlabel('残差')
plt.ylabel('频数')
plt.show()

# 9. 多步滚动预测（未来30天）
future_steps = 30
last_seq = data_scaled[-seq_len:].copy()
future_preds = []

for _ in range(future_steps):
    seq_t = torch.tensor(last_seq[np.newaxis,:,:],dtype=torch.float32)
    lstm_p = lstm_model(seq_t).detach().numpy()
    transformer_p = transformer_model(seq_t).detach().numpy()
    tcn_p = tcn_model(seq_t).detach().numpy()
    stacked_X = np.hstack([lstm_p, transformer_p, tcn_p])
    res_p = xgb_model.predict(stacked_X).reshape(-1,1)
    pred = (lstm_p + transformer_p + tcn_p)/3 + res_p
    future_preds.append(pred[0,0])
    # 更新序列（只更新第一列 y，其它特征使用0）
    new_row = np.zeros(data_scaled.shape[1])
    new_row[0] = pred
    last_seq = np.vstack([last_seq[1:], new_row])

future_preds_inv = scaler.inverse_transform(np.hstack([np.array(future_preds).reshape(-1,1), np.zeros((future_steps,4))]))[:,0]
future_dates = pd.date_range(pd.to_datetime(df['Date'].iloc[-1])+pd.Timedelta(days=1), periods=future_steps)

plt.figure(figsize=(14,5))
plt.plot(df['Date'], df['Open'], label='历史真实值', color='#FF6347')
plt.plot(future_dates, future_preds_inv, label='未来30天预测', color='#00CED1')
plt.title('未来30天滚动预测')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.show()