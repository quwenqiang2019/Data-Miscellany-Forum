import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import random


class woa():
    # 初始化
    def __init__(self, LB, UB, dim=4, b=1, whale_num=20, max_iter=500):
        self.LB = LB
        self.UB = UB
        self.dim = dim
        self.whale_num = whale_num
        self.max_iter = max_iter
        self.b = b
        # Initialize the locations of whale
        self.X = np.random.uniform(0, 1, (whale_num, dim)) * (UB - LB) + LB

        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(max_iter)
        self.gBest_X = np.zeros(dim)

    # 适应度函数 max_depth，min_samples_split，min_samples_leaf，max_leaf_nodes
    def fitFunc(self, para):
        # 建立模型
        mse = training(para)
        return mse
        # 优化模块

    def opt(self):
        t = 0
        while t < self.max_iter:
            print('At iteration: ' + str(t))
            for i in range(self.whale_num):
                # 防止X溢出
                self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)  # Check boundries
                fitness = self.fitFunc(self.X[i, :])
                # Update the gBest_score and gBest_X
                if fitness <= self.gBest_score:
                    self.gBest_score = fitness
                    self.gBest_X = self.X[i, :].copy()
            print('self.gBest_score: ', self.gBest_score)
            print('self.gBest_X: ', self.gBest_X)
            a = 2 * (self.max_iter - t) / self.max_iter
            # Update the location of whales
            for i in range(self.whale_num):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                A = 2 * a * R1 - a
                C = 2 * R2
                l = 2 * np.random.uniform() - 1
                # 如果随机值大于0.5 就按以下算法更新X
                if p >= 0.5:
                    D = abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gBest_X
                else:
                    # 如果随机值小于0.5 就按以下算法更新X
                    if abs(A) < 1:
                        D = abs(C * self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A * D
                    else:
                        rand_index = np.random.randint(low=0, high=self.whale_num)
                        X_rand = self.X[rand_index, :]
                        D = abs(C * X_rand - self.X[i, :])
                        self.X[i, :] = X_rand - A * D
            self.gBest_curve[t] = self.gBest_score
            t += 1
        return self.gBest_curve, self.gBest_X


    # 定义滑动窗口函数
def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0:data.shape[1]])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)


def build_model(X_train, neurons1, neurons2, dropout):
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
    model1.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    return model1



def training(X):
    neurons1 = int(X[0])
    neurons2 = int(X[1])
    dropout = round(X[2], 6)
    batch_size = int(X[3])
    print(X)
    model = build_model(X_train, neurons1, neurons2, dropout)
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=22, restore_best_weights=True)])

    pred = model.predict(X_test)
    temp_mse = mean_squared_error(y_test, pred)
    return temp_mse





if __name__ == "__main__":
    '''
    神经网络第一层神经元个数
    神经网络第二层神经元个数
    dropout比率
    batch_size
    '''
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Training Data')
    plt.plot(test_data, label='Testing Data')
    plt.xlabel('Year')
    plt.ylabel('Passenger Count')
    plt.title('International Airline Passengers - Training and Testing Data')
    plt.legend()
    plt.show()


    scaler = MinMaxScaler()
    train_data_scaler = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_data_scaler = scaler.transform(test_data.values.reshape(-1, 1))

    # 定义滑动窗口大小
    window_size = 1

    # 创建滑动窗口数据集
    X_train, y_train = create_sliding_windows(train_data_scaler, window_size)
    X_test, y_test = create_sliding_windows(test_data_scaler, window_size)

    # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))


    # 参数的上限
    UB = np.array([20, 100, 0.01, 36])
    # 参数的下限
    LB = np.array([5, 20, 0.00001, 5])

    # 开始优化===========主程序================
    Max_iter = 3  # 迭代次数
    dim = 4  # 鲸鱼的维度
    SearchAgents_no = 2  # 寻值的鲸鱼的数量
    fitnessCurve, para = woa(LB, UB, dim=dim, whale_num=SearchAgents_no, max_iter=Max_iter).opt()
    print('best_params is ', para)

    # 训练模型  使用WOA找到的最好的神经元个数
    neurons1 = int(para[0])
    neurons2 = int(para[1])
    dropout = para[2]
    batch_size = int(para[3])

    model = build_model(X_train, neurons1, neurons2, dropout)
    history1 = model.fit(X_train, y_train, epochs=150, batch_size=batch_size, validation_split=0.2, verbose=1,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)])


    # 使用 LSTM 模型进行预测
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)


    # 反归一化预测结果
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

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
    plt.plot(list(train_data.index)[window_size:train_size], train_predictions, label='Training Predictions')
    plt.plot(list(test_data.index)[-(len(test_data)-window_size):], test_predictions, label='Testing Predictions')
    plt.xlabel('Year')
    plt.ylabel('Passenger Count')
    plt.title('International Airline Passengers - Actual vs Predicted')
    plt.legend()
    plt.show()