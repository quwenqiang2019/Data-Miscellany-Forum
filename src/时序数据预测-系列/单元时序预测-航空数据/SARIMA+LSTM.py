import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
data = pd.read_csv('international-airline-passengers.csv')
# 将日期列转换为日期时间类型
data['Month'] = pd.to_datetime(data['Month'])
# 将日期列设置为索引
data.set_index('Month', inplace=True)


# 拆分数据集为训练集和测试集
train_data = data.iloc[:-12]
test_data = data.iloc[-12:]


# 拟合 SARIMA 模型并提取残差
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()
residuals = model_fit.resid
print(residuals)
# 进行预测
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
# 进行预测
sarima_predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
sarima_residuals = test_data - sarima_predictions
print(sarima_residuals)


# 构建 LSTM 模型
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 准备训练数据
train_size = int(len(residuals) * 0.8)
train_residuals = residuals[:train_size]
test_residuals = residuals[train_size:]

# 将数据转换为适合 LSTM 模型的输入格式
def create_lstm_dataset(data, lookback=1):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        Y.append(data[i + lookback])
    return np.array(X), np.array(Y)

lookback = 1
train_X, train_Y = create_lstm_dataset(train_residuals, lookback)
test_X, test_Y = create_lstm_dataset(test_residuals, lookback)

# 创建和拟合 LSTM 模型
lstm_model = create_lstm_model()
lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# 使用 LSTM 模型进行残差预测
train_residuals_prediction = lstm_model.predict(train_X)
test_residuals_prediction = lstm_model.predict(test_X)


scaler = MinMaxScaler()
# 反转缩放
train_residuals_prediction = scaler.inverse_transform(train_residuals_prediction)
test_residuals_prediction = scaler.inverse_transform(test_residuals_prediction)

# SARIMA 模型的预测值
sarima_predictions = sarima_results.fittedvalues
print(sarima_predictions, len(sarima_predictions))   # 144
print(test_residuals_prediction, len(test_residuals_prediction))  # 28


# # 将 SARIMA 模型的预测值与 LSTM 模型的预测残差值相加得到乘客量的预测值
# train_predictions = sarima_predictions[:train_size] + np.concatenate(train_residuals_prediction, axis=0).reshape(-1, 1)
print(sarima_predictions[train_size:])
print(len(sarima_predictions[train_size:]))
test_predictions = sarima_predictions[train_size+1:] + test_residuals_prediction
print(test_predictions, len(test_predictions))
#
# # 逆缩放预测值
# train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
# test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
#
# # 输出预测结果
# print("Train Passenger Predictions:", train_predictions)
# print("Test Passenger Predictions:", test_predictions)
