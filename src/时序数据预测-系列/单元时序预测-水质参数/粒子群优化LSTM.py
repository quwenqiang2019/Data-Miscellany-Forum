import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
def process_data():
    # 读取数据集
    data = pd.read_csv('international-airline-passengers.csv')
    # 将日期列转换为日期时间类型
    data['Month'] = pd.to_datetime(data['Month'])
    # 将日期列设置为索引
    data.set_index('Month', inplace=True)

    # 划分训练集和测试集
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 绘制训练集和测试集的折线图
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
    def create_sliding_windows(data, window_size):
        X, Y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            Y.append(data[i + window_size])
        return np.array(X), np.array(Y)

    # 定义滑动窗口大小
    window_size = 12

    # 创建滑动窗口数据集
    X_train, Y_train = create_sliding_windows(train_data_scaler, window_size)
    X_test, Y_test = create_sliding_windows(test_data_scaler, window_size)

    # # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    # X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))
    return X_train, y_train, X_test, y_test


def build_model(neurons1, neurons2, dropout):
    X_train, y_train, X_test, y_test = process_data()
    # X_train, y_train = create_dataset(X_train, y_train, steps)
    # X_test, y_test = create_dataset(X_test, y_test, steps)
    nb_features = X_train.shape[2]
    input1 = X_train.shape[1]
    model1 = Sequential()
    model1.add(LSTM(
        input_shape=(input1, nb_features),
        units=neurons1,
        return_sequences=True))
    model1.add(Dropout(dropout))

    model1.add(LSTM(
        units=neurons2,
        return_sequences=False))
    model1.add(Dropout(dropout))

    model1.add(Dense(units=1))
    model1.add(Activation("linear"))
    model1.compile(loss='mse', optimizer='Adam', metrics='mae')
    return model1, X_train, y_train, X_test, y_test







'''
神经网络第一层神经元个数
神经网络第二层神经元个数
dropout比率
batch_size
'''
'''
 神经网络第一层神经元个数
 神经网络第二层神经元个数
 dropout比率
 batch_size
 '''
UP = [51, 6, 0.055, 9]
DOWN = [50, 5, 0.05, 8]
NGEN = 100
popsize = 100
parameters = [NGEN, popsize, DOWN, UP]
# 开始优化
aco = ACO(parameters)
aco.main()

# 训练模型  使用ssa找到的最好的神经元个数
neurons1 = int(aco.g_best[0])
neurons2 = int(aco.g_best[1])
dropout = aco.g_best[2]
batch_size = int(aco.g_best[3])
# neurons1 = 64
# neurons2 = 64
# dropout = 0.01
# batch_size = 32
model, X_train, y_train, X_test, y_test = build_model(neurons1, neurons2, dropout)
history1 = model.fit(X_train, y_train, epochs=150, batch_size=batch_size, validation_split=0.2, verbose=1,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)])
# 测试集预测
y_score = model.predict(X_test)
# 反归一化
y_score = scaler.inverse_transform(y_score.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

print("==========evaluation==============\n")
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import math

MAE = mean_absolute_error(y_test, y_score)
print('MAE: %.4f ' % MAE)
RMSE = math.sqrt(mean_squared_error(y_test, y_score))
print('RMSE: %.4f ' % (RMSE))
