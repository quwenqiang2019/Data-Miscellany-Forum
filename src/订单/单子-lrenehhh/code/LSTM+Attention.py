import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import LSTM, Dense, Attention, Input, Permute, Multiply, Dot, Activation, MultiHeadAttention, Flatten

# 读取数据集
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_excel(os.path.join(base_dir, 'data', 'data1.xlsx'))[['month', 'flu_rate']]
# 将日期列转换为日期时间类型
data['month'] = pd.to_datetime(data['month'])
# 将日期列设置为索引
data.set_index('month', inplace=True)

# 划分训练集和测试集
# train_size = int(len(data) * 0.8)
train_size = len(data)-12
train_data = data[:train_size]
test_data = data[train_size:]

# 绘制训练集和测试集的折线图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'LSTM-Attention1.png'))
plt.show()

# 将数据归一化到 0~1 范围
scaler = MinMaxScaler()
train_data_scaler = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_data_scaler = scaler.transform(test_data.values.reshape(-1, 1))

# 定义滑动窗口函数
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

np.random.seed(7)

# 定义滑动窗口大小
look_back = 1

# 创建滑动窗口数据集
X_train, Y_train = create_dataset(train_data_scaler, look_back)
X_test, Y_test = create_dataset(test_data_scaler, look_back)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
X_train = np.reshape(X_train, (X_train.shape[0],  X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# 构建 LSTM 模型
inputs = Input(shape=(look_back, 1))
lstm = LSTM(128, return_sequences=True)(inputs)
attention = Attention()([lstm, lstm])
attention = Flatten()(attention)
output = Dense(1)(attention)
model = Model(inputs=inputs, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=500, batch_size=1, verbose=2)


# 使用 LSTM 模型进行预测
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = train_predictions.reshape(-1, 1)
test_predictions = test_predictions.reshape(-1, 1)
print(train_predictions)
print(train_predictions.shape)

# 反归一化预测结果
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
print(test_predictions)

# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_data, label='Actual')
plt.plot(list(test_data.index)[-len(test_predictions):], test_predictions, label='Predicted')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.title('Actual vs Predicted')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'LSTM-Attention2.png'))
plt.show()

# 绘制原始数据、训练集预测结果和测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(data, label='Actual')
plt.plot(list(train_data.index)[look_back:train_size], train_predictions, label='Training Predictions')
plt.plot(list(test_data.index)[-(len(test_data)-look_back):], test_predictions, label='Testing Predictions')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'LSTM-Attention3.png'))
plt.show()
