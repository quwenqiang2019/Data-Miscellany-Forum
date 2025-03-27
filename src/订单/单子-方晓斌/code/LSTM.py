import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
import tensorflow as tf
import random
import os
import seaborn as sns
'''
LSTM模型训练中的一些操作（如参数初始化、数据分割等）具有随机性，这会导致每次训练后的模型表现有所不同。
解决方案：
设置随机种子： 可以通过设定全局和框架内的随机种子来固定随机性，以确保每次实验的结果一致。
'''
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# 读取数据集
filename = '56669_2020-2024.xlsx'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', "gcdata_zd", filename)))
data = data[['datetime', 'obv_24h']]
print(data)

# 划分训练集和测试集
train_data = data[(data['datetime'] >= '2020-01-01') & (data['datetime'] < '2024-01-01')]
train_data.set_index('datetime', inplace=True)
test_data = data[(data['datetime'] >= '2024-01-01')]
test_data.set_index('datetime', inplace=True)
train_size = int(len(train_data))

# 绘制训练集和测试集的折线图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'Simsun'], size=12)
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.xlabel('Year')
plt.ylabel('obv_24h')
plt.title('obv_24h - Training and Testing Data')
plt.legend()
plt.show()

# 将数据归一化到 0~1 范围
scaler = MinMaxScaler()
train_data_scaler = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_data_scaler = scaler.transform(test_data.values.reshape(-1, 1))

# 定义滑动窗口函数
def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0:data.shape[1]])
        Y.append(data[i+window_size,0])
    return np.array(X), np.array(Y)

# 定义滑动窗口大小
window_size = 3

# 创建滑动窗口数据集
X_train, Y_train = create_sliding_windows(train_data_scaler, window_size)
X_test, Y_test = create_sliding_windows(test_data_scaler, window_size)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))


# 构建 LSTM 模型
# input = Input(shape=(window_size, 1))
# lstm = LSTM(units=50, activation='relu')(input)
# output = Dense(units=1)(lstm)
# model = Model(inputs=input, outputs=output)


# # 构建多层 LSTM 模型
model = Sequential()
model.add(LSTM(50, input_shape=(window_size, 1), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, batch_size=32)
model.save(os.path.join(base_dir, 'models', '56669'),save_format='tf')

# 使用 LSTM 模型进行预测
# model = tf.keras.models.load_model(os.path.join(base_dir, 'models', '56669'))
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 反归一化预测结果
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# 绘制训练集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Actual')
plt.plot(list(train_data.index)[-len(train_predictions):], train_predictions, label='Predicted')
plt.xlabel('Month')
plt.ylabel('obv_24h')
plt.title('Actual vs Predicted-Train Dataset')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'lstm_pred_train.jpg'), bbox_inches='tight', dpi = 600)
plt.show()

# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_data, label='Actual')
plt.plot(list(test_data.index)[-len(test_predictions):], test_predictions, label='Predicted')
plt.xlabel('Month')
plt.ylabel('obv_24h')
plt.title('Actual vs Predicted-Test Dataset')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'lstm_pred_test.jpg'), bbox_inches='tight', dpi = 600)
plt.show()

timestamps = list(test_data.index)[-len(test_predictions):]
data_pred =  pd.DataFrame({'datetime': timestamps, 'obv_24': test_predictions.flatten().tolist()})
print(data_pred) # 362


# 读取数据集
filename = '56669_20230517-20241231.xlsx'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', "qpe_zd", filename)))
data = data[['datetime', 'qpe24']]
data = data[(data['datetime'] >= '2024-01-04') & (data['datetime'] <= '2024-12-30')]
print(data)  #326

merged_df = pd.merge(data_pred, data, on='datetime', how='left')
merged_df.to_excel(os.path.join(base_dir, 'result', 'result_compare.xlsx'))

# 绘制预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_data, label='Actual-obv_24h')
plt.plot(list(test_data.index)[-len(test_predictions):], test_predictions, label='Predicted-obv_24h')
plt.plot(data['datetime'], data['qpe24'], label='qpe24')
plt.xlabel('Month')
plt.title('Actual vs Predicted')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'lstm_pred_compare.jpg'), bbox_inches='tight', dpi = 600)
plt.show()