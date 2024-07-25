import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping
import random
import time
import seaborn as sns



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
        epochs=22,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=22, restore_best_weights=True)])

    pred = model.predict(X_test)
    le = len(pred)
    y_t = y_test.reshape(-1, 1)
    return pred, le, y_t



def function(ps, test, le):
    ss = sum(((abs(test - ps)) / test) / le)
    return ss


def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0:data.shape[1]])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)



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

    # (1) PSO Parameters
    MAX_EPISODES = 2
    MAX_EP_STEPS = 2
    c1 = 2
    c2 = 2
    w = 0.5
    pN = 2  # 粒子数量

    # (2) LSTM Parameters
    dim = 4  # 搜索维度
    X = np.zeros((pN, dim) )  # 所有粒子的位置和速度
    V = np.zeros((pN, dim))
    pbest = np.zeros((pN, dim))  # 个体经历的最佳位置和全局最佳位置
    gbest = np.zeros(dim)
    p_fit = np.zeros(pN)  # 每个个体的历史最佳适应值
    print(p_fit.shape)
    print(p_fit.shape)
    t1 = time.time()

    '''
    神经网络第一层神经元个数
    神经网络第二层神经元个数
    dropout比率
    batch_size
    '''
    UP = [150, 15, 0.5, 16]
    DOWN = [50, 5, 0.05, 8]

    # (4) 开始搜索
    for i_episode in range(MAX_EPISODES):
        """初始化s"""
        random.seed(8)
        fit = -1e5  # 全局最佳适应值
        # 初始粒子适应度计算
        print("计算初始全局最优")
        for i in range(pN):
            for j in range(dim):
                V[i][j] = random.uniform(0, 1)
                if j == 2:
                    X[i][j] = random.uniform(DOWN[j], UP[j])
                else:
                    X[i][j] = round(random.randint(DOWN[j], UP[j]), 0)
            pbest[i] = X[i]
            le, pred, y_t = training(X[i])
            NN = 1
            # 计算适应值
            tmp = function(pred, y_t, le)
            p_fit[i] = tmp
            if tmp > fit:
                fit = tmp
                gbest = X[i]
        print("初始全局最优参数：{:}".format(gbest))

        fitness = []  # 适应度函数
        for j in range(MAX_EP_STEPS):
            fit2 = []
            plt.title("第{}次迭代".format(i_episode))
            for i in range(pN):
                le, pred, y_t = training(X[i])
                temp = function(pred, y_t, le)
                fit2.append(temp / 1000)
                if temp > p_fit[i]:  # 更新个体最优
                    p_fit[i] = temp
                    pbest[i] = X[i]
                    if p_fit[i] > fit:  # 更新全局最优
                        gbest = X[i]
                        fit = p_fit[i]
            print("搜索步数：{:}".format(j))
            print("个体最优参数：{:}".format(pbest))
            print("全局最优参数：{:}".format(gbest))
            for i in range(pN):
                V[i] = w * V[i] + c1 * random.uniform(0, 1) * (pbest[i] - X[i]) + c2 * random.uniform(0, 1) * (
                        gbest - X[i])
                ww = 1
                for k in range(dim):
                    if DOWN[k] < X[i][k] + V[i][k] < UP[k]:
                        continue
                    else:
                        ww = 0
                X[i] = X[i] + V[i] * ww
            fitness.append(fit)
        # 画适应度的图
        plt.plot(fitness, label='Fitness-PSO')
        plt.legend(loc='best')
        plt.title('Fitness PSO')
        plt.xlabel('MAX_EP_STEPS')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.show()

    print('Running time: ', time.time() - t1)

    # 训练模型  使用PSO找到的最好的神经元个数
    neurons1 = int(gbest[0])
    neurons2 = int(gbest[1])
    dropout = gbest[2]
    batch_size = int(gbest[3])

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
    plt.plot(list(test_data.index)[-(len(test_data) - window_size):], test_predictions, label='Testing Predictions')
    plt.xlabel('Year')
    plt.ylabel('Passenger Count')
    plt.title('International Airline Passengers - Actual vs Predicted')
    plt.legend()
    plt.show()