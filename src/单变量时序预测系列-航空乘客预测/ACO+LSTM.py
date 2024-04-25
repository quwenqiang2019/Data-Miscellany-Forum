import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping



class ACO:
    def __init__(self, parameters):
        """
        Ant Colony Optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = len(parameters[2])  # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有蚂蚁的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局蚂蚁最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            fit = self.fitness(self.pop_x[i])
            if fit > temp:
                self.g_best = self.pop_x[i]
                temp = fit

    def fitness(self, ind_var):
        """
        个体适应值计算
        """
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]
        x4 = ind_var[3]
        y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
        return y

    def update_operator(self, gen, t, t_max):
        """
        更新算子：根据概率更新下一时刻的位置
        """
        rou = 0.8  # 信息素挥发系数
        Q = 1  # 信息释放总量
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.var_num):
                pi[i] = (t_max - t[i]) / t_max
                # 更新位置
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
                            self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保护
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新t值
            t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])
            # 更新全局最优值
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t

    def main(self):
        popobj = []
        best = np.zeros((1, self.var_num))[0]
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop_x))),
                                               np.max(np.array(list(map(self.fitness, self.pop_x)))))
            else:
                tmax, t = self.update_operator(gen, t, tmax)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen)))
            print(self.g_best)
            print(self.fitness(self.g_best))
            if self.fitness(self.g_best) > self.fitness(best):
                best = self.g_best.copy()
            print('最好的位置：{}'.format(best))
            print('最大的函数值：{}'.format(self.fitness(best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(1, self.NGEN + 1)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()

def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size])
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

# 将数据归一化到 0~1 范围
scaler = MinMaxScaler()
train_data_scaler = scaler.fit_transform(train_data.values.reshape(-1, 1))
print(train_data_scaler)
test_data_scaler = scaler.transform(test_data.values.reshape(-1, 1))
print(test_data_scaler)

# 定义滑动窗口函数
def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0:data.shape[1]])
        Y.append(data[i+window_size,0])
    return np.array(X), np.array(Y)



# 定义滑动窗口大小
window_size = 1

# 创建滑动窗口数据集
X_train, Y_train = create_sliding_windows(train_data_scaler, window_size)
X_test, Y_test = create_sliding_windows(test_data_scaler, window_size)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))



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

model = build_model(X_train, neurons1, neurons2, dropout)
history1 = model.fit(X_train, Y_train, epochs=150, batch_size=batch_size, validation_split=0.2, verbose=1,
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