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


class SSA():
    def __init__(self, func, n_dim=None, pop_size=20, max_iter=50, lb=-512, ub=512, verbose=False):
        self.func = func
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.pop = pop_size  # number of particles
        P_percent = 0.2  # # 生产者的人口规模占总人口规模的20%
        D_percent = 0.1  # 预警者的人口规模占总人口规模的10%
        self.pNum = round(self.pop * P_percent)  # 生产者的人口规模占总人口规模的20%
        self.warn = round(self.pop * D_percent)  # 预警者的人口规模占总人口规模的10%

        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))

        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_pbest()
        self.update_gbest()
        #
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        self.idx_max = 0
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def cal_y(self, start, end):
        # calculate y for every x in X
        for i in range(start, end):
            self.Y[i] = self.func(self.X[i])
        # return self.Y

    def update_pbest(self):
        '''
        personal best
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]

    def update_gbest(self):
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def find_worst(self):
        self.idx_max = self.Y.index(max(self.Y))
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def update_finder(self):
        r2 = np.random.rand(1)  # 预警值
        self.idx = sorted(enumerate(self.Y), key=lambda x: x[1])
        self.idx = [self.idx[i][0] for i in range(len(self.idx))]
        # 这一部位为发现者（探索者）的位置更新
        if r2 < 0.8:  # 预警值较小，说明没有捕食者出现
            for i in range(self.pNum):
                r1 = np.random.rand(1)
                self.X[self.idx[i], :] = self.X[self.idx[i], :] * np.exp(-(i) / (r1 * self.max_iter))  # 对自变量做一个随机变换
                self.X = np.clip(self.X, self.lb, self.ub)  # 对超过边界的变量进行去除
                # X[idx[i], :] = Bounds(X[idx[i], :], lb, ub)  # 对超过边界的变量进行去除
                # fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])  # 算新的适应度值
        elif r2 >= 0.8:  # 预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(self.pNum):
                Q = np.random.rand(1)  # 也可以替换成  np.random.normal(loc=0, scale=1.0, size=1)
                self.X[self.idx[i], :] = self.X[self.idx[i], :] + Q * np.ones(
                    (1, self.n_dim))  # Q是服从正态分布的随机数。L表示一个1×d的矩阵
                self.X = np.clip(self.X, self.lb, self.ub)  # 对超过边界的变量进行去除
                # X[idx[i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                # fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])
        self.cal_y(0, self.pNum)

    def update_follower(self):
        #  这一部位为加入者（追随者）的位置更新
        for ii in range(self.pop - self.pNum):
            i = ii + self.pNum
            A = np.floor(np.random.rand(1, self.n_dim) * 2) * 2 - 1
            best_idx = self.Y[0:self.pNum].index(min(self.Y[0:self.pNum]))
            bestXX = self.X[best_idx, :]
            if i > self.pop / 2:
                Q = np.random.rand(1)
                self.X[self.idx[i], :] = Q * np.exp((self.x_max - self.X[self.idx[i], :]) / np.square(i))
            else:
                self.X[self.idx[i], :] = bestXX + np.dot(np.abs(self.X[self.idx[i], :] - bestXX),
                                                         1 / (A.T * np.dot(A, A.T))) * np.ones((1, self.n_dim))
        self.X = np.clip(self.X, self.lb, self.ub)  # 对超过边界的变量进行去除
        # X[self.idx[i],:] = Bounds(X[self.idx[i],lb,ub)
        # fit[self.idx[i],0] = func(X[self.idx[i], :])
        self.cal_y(self.pNum, self.pop)

    def detect(self):
        arrc = np.arange(self.pop)
        c = np.random.permutation(arrc)  # 随机排列序列
        b = [self.idx[i] for i in c[0: self.warn]]
        e = 10e-10
        for j in range(len(b)):
            if self.Y[b[j]] > self.gbest_y:
                self.X[b[j], :] = self.gbest_y + np.random.rand(1, self.n_dim) * np.abs(self.X[b[j], :] - self.gbest_y)
            else:
                self.X[b[j], :] = self.X[b[j], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    self.X[b[j], :] - self.x_max) / (self.func(self.X[b[j]]) - self.y_max + e)
            # X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            # fit[sortIndex[0, b[j]], 0] = func(X[sortIndex[0, b[j]]])
            self.X = np.clip(self.X, self.lb, self.ub)  # 对超过边界的变量进行去除
            self.Y[b[j]] = self.func(self.X[b[j]])

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_finder()  # 更新发现者位置
            self.find_worst()  # 取出最大的适应度值和最差适应度的X
            self.update_follower()  # 更新跟随着位置
            self.update_pbest()
            self.update_gbest()
            self.detect()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        return self.best_x, self.best_y


    # 定义滑动窗口函数
def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0:data.shape[1]])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

def process_data(train_data, test_data):
    # 将数据归一化到 0~1 范围
    # scaler = MinMaxScaler()
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

    return X_train, y_train, X_test, y_test


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


    UP = [51, 6, 0.055, 9]
    DOWN = [50, 5, 0.05, 8]

    # 开始优化
    ssa = SSA(training, n_dim=4, pop_size=22, max_iter=1, lb=DOWN, ub=UP)
    ssa.run()
    print('best_params is ', ssa.gbest_x)
    print('best_precision is', 1 - ssa.gbest_y)

    # 训练模型  使用ssa找到的最好的神经元个数
    neurons1 = int(ssa.gbest_x[0])
    neurons2 = int(ssa.gbest_x[1])
    dropout = ssa.gbest_x[2]
    batch_size = int(ssa.gbest_x[3])

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