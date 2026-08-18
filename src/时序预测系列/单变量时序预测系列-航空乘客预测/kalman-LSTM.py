# // Below is the code of /src/单变量时序预测系列-航空乘客预测/kalman-LSTM.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns


# LSTM 网络定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 预测并应用 Kalman 滤波
def apply_kalman_filter(predictions):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(predictions, n_iter=5)
    kalman_smoothed, _ = kf.smooth(predictions)
    return kalman_smoothed


# 主流程
if __name__ == "__main__":
    # 读取数据集
    data = pd.DataFrame(pd.read_csv('data.csv'))
    print(data)
    # 将日期列转换为日期时间类型
    data['Month'] = pd.to_datetime(data['Month'])
    time = data['Month'].values
    signal = data['Passengers'].values

    # 准备数据
    seq_length = 7
    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1))
    X, y = [], []
    for i in range(len(signal_scaled) - seq_length):
        X.append(signal_scaled[i:i+seq_length])
        y.append(signal_scaled[i+seq_length])

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    # 模型训练
    num_epochs = 100
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    # 预测并应用 Kalman 滤波
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    X_tensor = X_tensor.to('cuda:0')
    lstm_predictions = model(X_tensor).detach().cpu().numpy()
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    kalman_predictions = apply_kalman_filter(lstm_predictions)

    # 画图分析
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.plot(time, signal, label='Original Signal', color='blue', alpha=0.6)
    plt.plot(time[len(time)-len(lstm_predictions):], lstm_predictions, label='LSTM Prediction', color='red')
    plt.plot(time[len(time)-len(kalman_predictions):], kalman_predictions, label='Kalman Filtered', color='green')
    plt.legend()
    plt.title('LSTM + Kalman Filter Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()