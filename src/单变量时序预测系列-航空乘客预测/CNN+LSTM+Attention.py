import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import *
import seaborn as sns

# 读取数据集
data = pd.read_csv('data.csv')
# 将日期列转换为日期时间类型
data['Month'] = pd.to_datetime(data['Month'])
# 将日期列设置为索引
data.set_index('Month', inplace=True)

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 绘制训练集和测试集的折线图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - Training and Testing Data')
plt.legend()
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
look_back = 3

# 创建滑动窗口数据集
X_train, Y_train = create_dataset(train_data_scaler, look_back)
X_test, Y_test = create_dataset(test_data_scaler, look_back)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
X_train = np.reshape(X_train, (X_train.shape[0],  X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 注意力机制
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    # 根据给定的模式(dim)置换输入的维度  例如(2,1)即置换输入的第1和第2个维度
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # Layer that multiplies (element-wise) a list of inputs.
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_block(inputs,time_step):
    # batch_size, time_steps, lstm_units -> batch_size, lstm_units, time_steps
    a = Permute((2, 1))(inputs)
    # batch_size, lstm_units, time_steps -> batch_size, lstm_units, time_steps
    a = Dense(time_step, activation='softmax')(a)#和步长有关
    # batch_size, lstm_units, time_steps -> batch_size, time_steps, lstm_units
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # 相当于获得每一个step中，每个特征的权重
    output_attention_mul = concatenate([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# 构建 LSTM 模型
lstm_units = 50
dropout = 0.01
inputs=Input(shape=(look_back, 1))
my_model=Conv1D(filters = lstm_units, kernel_size = 1, activation = 'sigmoid')(inputs)#卷积层
my_model=Dropout(dropout)(my_model)#droupout层
my_model=LSTM(lstm_units, activation='tanh', return_sequences=True)(my_model)      #LSTM层
print(my_model)
# attention = Attention()([my_model, my_model])
# attention = attention_3d_block(my_model)
attention = attention_block(my_model, look_back)
attention = Flatten()(attention)
outputs = Dense(1, activation='tanh')(attention)
my_model = Model(inputs=inputs, outputs=outputs)
my_model.compile(loss='mean_squared_error', optimizer='adam')
my_model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=2)


# 使用 LSTM 模型进行预测
train_predictions = my_model.predict(X_train)
test_predictions = my_model.predict(X_test)

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
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# 绘制原始数据、训练集预测结果和测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(data, label='Actual')
plt.plot(list(train_data.index)[look_back:train_size], train_predictions, label='Training Predictions')
plt.plot(list(test_data.index)[-(len(test_data)-look_back):], test_predictions, label='Testing Predictions')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - Actual vs Predicted')
plt.legend()
plt.show()
