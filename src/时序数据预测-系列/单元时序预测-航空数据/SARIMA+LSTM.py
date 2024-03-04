import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
data = pd.read_csv('international-airline-passengers.csv')
data['Month'] = pd.to_datetime(data['Month'])   # 将日期列转换为日期时间类型
data.set_index('Month', inplace=True)    # 将日期列设置为索引
data = data['Passengers'].values

# 拆分数据集为训练集和测试集
# train_size = int(len(data) * 0.8)
train_size = len(data) - 12
train_data = data[:train_size]
test_data = data[train_size:]
print(train_data, len(train_data))

# 拟合 SARIMA 模型并提取残差
sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()
sarima_train_predictions = sarima_model_fit.predict(start=0, end=train_size-1)
# 训练集预测的第一个值是0
sarima_train_predictions[0] = train_data[0]
print(sarima_train_predictions, len(sarima_train_predictions))

# 计算残差序列
train_residuals = train_data - sarima_train_predictions
print(train_residuals, len(train_residuals))

# 归一化残差序列
scaler = MinMaxScaler()
scaled_train_residuals = scaler.fit_transform(train_residuals.reshape(-1, 1))

# LSTM模型训练和预测
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 1
train_X, train_Y = create_dataset(scaled_train_residuals, look_back)

lstm_model = Sequential()
lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)

# LSTM模型预测整个训练集的残差值
lstm_train_residuals = lstm_model.predict(train_X)
lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
print(lstm_train_residuals, len(lstm_train_residuals))    # look_back = 1，第一个残差无法预测

# SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
print("最终训练集的预测值:", train_predictions)

# 绘制训练集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(train_predictions, label='Predicted')
plt.plot(train_data[1:], label='Actual')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# SARIMA模型测试集预测值
sarima_test_predictions = sarima_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
print(sarima_test_predictions, len(sarima_test_predictions))

# 计算残差序列
sarima_test_residuals = test_data - sarima_test_predictions

# 归一化残差序列
scaled_test_residuals = scaler.transform(sarima_test_residuals.reshape(-1, 1))

# 构造残差数据集
test_X, test_Y = create_dataset(scaled_test_residuals, look_back)

# LSTM模型预测整个测试集的残差值
lstm_test_residuals = lstm_model.predict(test_X)
lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
print(lstm_test_residuals, len(lstm_test_residuals))

# SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
print("最终测试集的预测值:", test_predictions)

# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_predictions, label='Predicted')
plt.plot(test_data[1:], label='Actual')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()


