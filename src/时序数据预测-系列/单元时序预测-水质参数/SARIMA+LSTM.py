import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_excel('样点5.xlsx')
data = pd.DataFrame(data)
# 使用插值法填充缺失值
data['TSM值'] = data['TSM值'].interpolate()
# 将日期列转换为日期时间类型
data['日期'] = pd.to_datetime(data['日期'], format='%Y%m')


# # 将日期列设置为索引
data.set_index('日期', inplace=True)
# 构造规律的时间间隔
data = data.resample('MS').asfreq()
# 使用插值法填充缺失值
data['TSM值'] = data['TSM值'].interpolate()
dates = data.index

passengers = data['TSM值'].values

# 分割训练集和测试集
train_size = int(len(passengers) * 0.8)
train_data, test_data = passengers[:train_size], passengers[train_size:]

# SARIMA 模型训练和预测
sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit(disp=False)
sarima_predictions = sarima_model_fit.predict(start=train_size, end=train_size + len(test_data) - 1)

# LSTM 模型训练和预测
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data.reshape(-1, 1))

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # 序列长度
X_train, y_train = create_sequences(scaled_train_data, seq_length)

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

# 使用 LSTM 模型进行预测
lstm_predictions = []
current_batch = scaled_train_data[-seq_length:].reshape((1, seq_length, 1))
for i in range(len(test_data)):
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions.append(lstm_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[lstm_pred]], axis=1)

lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

# 组合模型预测结果
combined_predictions = 0.8*sarima_predictions + 0.2*lstm_predictions

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(test_data, combined_predictions))

# 绘制预测结果和实际值
plt.plot(dates[train_size:], test_data, label='Actual')
plt.plot(dates[train_size:], combined_predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('International Airline Passengers - SARIMA and LSTM')
plt.legend()
plt.show()

# 打印均方根误差
print('RMSE:', rmse)
