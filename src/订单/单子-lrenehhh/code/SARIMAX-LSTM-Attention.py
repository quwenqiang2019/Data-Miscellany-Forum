import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os

from keras.models import Model
from keras.layers import LSTM, Dense, Attention, Input, Flatten


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 读取数据集
df = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', 'data1.xlsx')))
data = df[['month', 'flu_rate']]

# # 生成一个非常接近于0的随机值列表
# replacement_values = np.random.uniform(0.01, 0.1, size=(data['flu_rate'] == 0).sum())
# # 将flu_rate列中的0替换为随机值
# data.loc[data['flu_rate'] == 0, 'flu_rate'] = replacement_values

data['month'] = pd.to_datetime(data['month'])   # 将日期列转换为日期时间类型
data.set_index('month', inplace=True)    # 将日期列设置为索引
data = data['flu_rate'].values

# 拆分数据集为训练集和测试集
# train_size = int(len(data) * 0.8)
train_size = len(data) - 12
train_data = data[:train_size]
test_data = data[train_size:]
print(train_data, len(train_data))

# 拟合 SARIMA 模型并提取残差
sarima_model = SARIMAX(train_data, exog=df[['temperature', 'rain']].values[:train_size], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
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

# 构造LSTM-Attention模型
inputs = Input(shape=(look_back, 1))
lstm = LSTM(128, return_sequences=True)(inputs)
attention = Attention()([lstm, lstm])
attention = Flatten()(attention)
output = Dense(1)(attention)
model = Model(inputs=inputs, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=50, batch_size=1, verbose=2)

# LSTM模型预测整个训练集的残差值
lstm_train_residuals = model.predict(train_X)
lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
print(lstm_train_residuals, len(lstm_train_residuals))    # look_back = 1，第一个残差无法预测

# SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
print("最终训练集的预测值:", train_predictions)

# 绘制训练集预测结果的折线图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.plot(train_predictions, label='Predicted')
plt.plot(train_data[1:], label='Actual')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.title('Actual vs Predicted')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'SARIMA-LSTM-Attention1.tif'))
plt.show()

# SARIMA模型测试集预测值
sarima_test_predictions = sarima_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, exog=df[['temperature', 'rain']].values[train_size:])
print(sarima_test_predictions, len(sarima_test_predictions))

# 计算残差序列
sarima_test_residuals = test_data - sarima_test_predictions

# 归一化残差序列
scaled_test_residuals = scaler.transform(sarima_test_residuals.reshape(-1, 1))

# 构造残差数据集
test_X, test_Y = create_dataset(scaled_test_residuals, look_back)

# LSTM模型预测整个测试集的残差值
lstm_test_residuals = model.predict(test_X)
lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
print(lstm_test_residuals, len(lstm_test_residuals))

# SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
print("最终测试集的预测值:", test_predictions)
# 绘制测试集预测结果的折线图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.plot(test_predictions, label='Predicted')
plt.plot(test_data[1:], label='Actual')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.title('Actual vs Predicted')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'SARIMA-LSTM-Attention2.tif'))
plt.show()


