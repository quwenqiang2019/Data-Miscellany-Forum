import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import math
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Multiply
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


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
        plt.title(f"{point}")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(1, self.NGEN + 1)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.savefig(f'result/{point}/aco.jpg', bbox_inches='tight', dpi=600)
        plt.show()

#=======================================异常值处理========================
def replace_outliers(series):
    # 计算均值和标准差
    mean = series.mean().values[0]
    std = series.std().values[0]
    # 设置阈值
    threshold = mean + 2 * std

    for date in series.index:
        year = date.year
        month = date.month

        if series.loc[date].values[0] > threshold:
            same_month_data = series[(series.index.year != year) & (series.index.month == month)]
            month_mean = same_month_data.mean()
            series.loc[date] = month_mean
    return series

#====================缺失值处理========================
def fill_missing_data(series):
    for date in series.index:
        year = date.year
        month = date.month

        if pd.isnull(series.loc[date].values[0]):
            same_month_data = series[(series.index.year != year) & (series.index.month == month)]
            month_mean = same_month_data.mean()
            series.loc[date] = month_mean
    return series



def data_preprocess(file):
    # =============================读取数据集===============================
    data = pd.read_excel(file)
    data = pd.DataFrame(data)
    # 将日期列转换为日期时间类型
    data['日期'] = pd.to_datetime(data['日期'], format='%Y%m')
    # 将日期列设置为索引
    data.set_index('日期', inplace=True)
    cols = list(data.columns)
    print(data)

    for i in cols:
        data[[i]] = replace_outliers(data[[i]])

    for i in cols:
        data[[i]] = fill_missing_data(data[[i]])
    # ===============构造规律的时间间隔========================
    data = data.resample('MS').asfreq()
    data = pd.DataFrame(data)

    return data

def seasonal_decompose(data):
    # sns.set_style('darkgrid')
    sns.set(font_scale=1.2)
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    # 时序数据分解
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(data)
    result.plot()
    plt.savefig(f'result/{point}/时序数据分解.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

def plot_acf(data):
    # ACF：自相关函数
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(data)
    plt.savefig(f'result/{point}/acf.jpg', bbox_inches='tight', dpi=600)
    plt.show()


def plot_pacf(data):
    # PACF：偏自相关函数
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(data)
    plt.savefig(f'result/{point}/pacf.jpg', bbox_inches='tight', dpi=600)
    plt.show()

def data_analysis(data):
    seasonal_decompose(data)
    plot_acf(data)
    plot_pacf(data)
    # 平稳性检验：Dickey-Fuller检验
    from statsmodels.tsa.stattools import adfuller
    adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(data)
    print('ADF test statistic:', adf)
    print('ADF p-values:', pval)
    print('ADF used number of lags:', usedlag)
    print('ADF number of observations:', nobs)
    print('ADF critical values:', crit_vals)
    print('ADF best information criterion: ', icbest)


# ============================================检验时间序列平稳性===========================================
# 方法一：绘制移动平均值和标准差
def TestStationaryPlot(df):
    rol_mean = df.rolling(window=12, center=False).mean()
    rol_std = df.rolling(window=12, center=False).std()

    # sns.set_style('darkgrid')
    sns.set(font_scale=1.2)
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(df, color='blue', label='Original')
    plt.plot(rol_mean, color='red', linestyle='-.', label='Moving Average')
    plt.plot(rol_std, color='black', linestyle='--', label='Standard Deviation')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel('Time', fontsize=14)
    plt.ylabel(f'{parameters}-{point}', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.title('Moving Average and Standard Deviation', fontsize=14)
    plt.savefig(f'result/{point}/移动平均值和标准差.jpg', bbox_inches='tight', dpi=600)
    plt.show(block=True)


#方法二：ADF检验
def TestStationaryAdfuller(df, cutoff = 0.01):
    df_test = adfuller(df, autolag = 'AIC')
    df_test_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in df_test[4].items():
        df_test_output['Critical Value (%s)'%key] = value
    print(df_test_output)

    if df_test[1] <= cutoff:
        print('拒绝原假设，即数据没有单位根,序列是平稳的。')
    else:
        print('不能拒绝原假设，即数据存在单位根,数据是非平稳序列。')



def data_split(data):
    # ========================== 划分数据集 ==================================
    train_size = len(data) - 15
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_data_key = train_data[f'{parameters}']
    train_data_fz = train_data.drop([f'{parameters}'], axis=1)
    test_data_key = test_data[f'{parameters}']
    test_data_fz = test_data.drop([f'{parameters}'], axis=1)

    # 绘制训练集和测试集的折线图
    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key, label='训练集')
    plt.plot(test_data_key, label='测试集')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'{point} - 训练集和测试集')
    plt.legend()
    plt.savefig(f'result/{point}/Training and Testing Data.jpg', bbox_inches='tight', dpi=600)
    plt.show()

    return train_data_key, train_data_fz, test_data_key, test_data_fz


def holt_winters(train_data_key, train_data_fz, test_data_key, test_data_fz):
    model = ExponentialSmoothing(train_data_key, trend="add", seasonal="add", seasonal_periods=12)
    model_fit = model.fit()
    predictions = model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])# 进行预测

    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    # sns.set(font_scale=1.2)
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False

    # 绘制训练集预测结果折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key.index, train_data_key, label='真实值')
    plt.plot(train_data_key.index, model_fit.fittedvalues, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'holt winters: {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/holt_winters_taian.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 绘制测试集预测结果折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key.index, test_data_key, label='真实值')
    plt.plot(predictions.index, predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'holt_winters: {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/holt_winters_test.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 计算误差
    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], model_fit.fittedvalues[1:]))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], predictions[1:]))
    trainScore2 = mean_absolute_error(train_data_key[1:], model_fit.fittedvalues[1:])
    testScore2 = mean_absolute_error(test_data_key[1:], predictions[1:])
    trainScore3 = r2_score(train_data_key[1:], model_fit.fittedvalues[1:])
    testScore3 = r2_score(test_data_key[1:], predictions[1:])
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], model_fit.fittedvalues[1:])
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], predictions[1:])

    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})


    df.to_excel(writer, sheet_name='holt_winters', index=False)

    return predictions


def sarima(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 拟合 SARIMA 模型
    model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    # 进行预测
    predictions = model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])

    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False


    # 绘制原始数据、训练集预测结果和测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key.index, train_data_key, label='真实值')
    plt.plot(train_data_key.index, model_fit.fittedvalues, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima: {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_taian.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key.index, test_data_key, label='真实值')
    plt.plot(predictions.index, predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima: {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_test.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 计算误差
    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], model_fit.fittedvalues[1:]))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], predictions[1:]))
    trainScore2 = mean_absolute_error(train_data_key[1:], model_fit.fittedvalues[1:])
    testScore2 = mean_absolute_error(test_data_key[1:], predictions[1:])
    trainScore3 = r2_score(train_data_key[1:], model_fit.fittedvalues[1:])
    testScore3 = r2_score(test_data_key[1:], predictions[1:])
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], model_fit.fittedvalues[1:])
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], predictions[1:])



    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})

    print(df)

    df.to_excel(writer, sheet_name='sarima', index=False)

    return predictions


def sarima_grid_search(data, parameters):
    # 首先定义 p、d、q 的参数值范围，这里取 0 - 2.
    p = d = q = range(0, 2)
    # 然后用itertools函数生成不同的参数组合
    pdq = list(itertools.product(p, d, q))
    # 同理处理季节周期性参数，也生成相应的多个组合
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    dic = {}

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data[f'{parameters}'],
                                                             order=param,
                                                             seasonal_order=param_seasonal,
                                                             enforce_stationarity=False,
                                                             enforce_invertibility=False)

                results = mod.fit()

                print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                dic.update({results.aic: [param, param_seasonal]})
            except:
                continue

    print(dic)
    # ===================================================SARIMA最优模型的参数===============================
    mod = sm.tsa.statespace.SARIMAX(data[parameters],
                                                 order=(1, 1, 1),
                                                 seasonal_order=(1, 1, 1, 12),
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)
    results = mod.fit()
    results.plot_diagnostics(figsize=(12, 10))
    plt.show()
    print(results.summary().tables[1])

def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0:data.shape[1]])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)


def lstm(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 将数据归一化到 0~1 范围
    scaler = MinMaxScaler()
    train_data_scaler = scaler.fit_transform(train_data_key.values.reshape(-1, 1))
    test_data_scaler = scaler.transform(test_data_key.values.reshape(-1, 1))

    # 定义滑动窗口大小
    window_size = 1

    # 创建滑动窗口数据集
    X_train, Y_train = create_sliding_windows(train_data_scaler, window_size)
    X_test, Y_test = create_sliding_windows(test_data_scaler, window_size)

    # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(60, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # 训练 LSTM 模型
    model.fit(X_train, Y_train, epochs=100, batch_size=32)

    # 使用 LSTM 模型进行预测
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # 反归一化预测结果
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)


    # # 绘制原始数据、训练集预测结果和测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key[window_size:], label='真实值')
    plt.plot(list(train_data_key.index)[-len(train_predictions):], train_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'lstm: {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/lstm_v1_train.jpg', bbox_inches='tight', dpi=600)
    plt.show()


    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key[window_size:], label='真实值')
    plt.plot(list(test_data_key.index)[-len(test_predictions):], test_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'lstm: {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/lstm_v1_test.jpg', bbox_inches='tight', dpi=600)
    plt.show()


 # 计算误差(预测精度)
    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], train_predictions))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], test_predictions))
    trainScore2 = mean_absolute_error(train_data_key[1:], train_predictions)
    testScore2 = mean_absolute_error(test_data_key[1:], test_predictions)
    trainScore3 = r2_score(train_data_key[1:], train_predictions)
    testScore3 = r2_score(test_data_key[1:], test_predictions)
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], train_predictions)
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], test_predictions)

    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})

    print(df)

    df.to_excel(writer, sheet_name='lstm_v1', index=False)

    return test_predictions

def holt_winters_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 拟合 SARIMA 模型并提取残差
    holt_winters_model = ExponentialSmoothing(train_data_key, trend="add", seasonal="add", seasonal_periods=12)
    holt_winters_model_fit = holt_winters_model.fit()
    holt_winters_train_predictions = holt_winters_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
    # 训练集预测的第一个值是0
    holt_winters_train_predictions[0] = train_data_key[0]
    # 计算残差序列
    train_residuals = train_data_key - holt_winters_train_predictions
    # 归一化残差序列
    scaler = MinMaxScaler()
    scaled_train_residuals = scaler.fit_transform(np.array(train_residuals).reshape(-1, 1))
    # LSTM模型训练和预测
    look_back = 1
    train_X, train_Y = create_sliding_windows(scaled_train_residuals, look_back)
    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)
    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = holt_winters_train_predictions[1:] + lstm_train_residuals.flatten()
    # 绘制训练集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key[1:], label='真实值')
    plt.plot(train_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'holt winters + lstm: {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/holt_winters_lstm_taian.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # SARIMA模型测试集预测值
    sarima_test_predictions = holt_winters_model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])
    # 计算残差序列
    sarima_test_residuals = test_data_key - sarima_test_predictions
    # 归一化残差序列
    scaled_test_residuals = scaler.transform(np.array(sarima_test_residuals).reshape(-1, 1))
    # 构造残差数据集
    test_X, test_Y = create_sliding_windows(scaled_test_residuals, look_back)
    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
    test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key[1:], label='真实值')
    plt.plot(test_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'holt winters + lstm: {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/holt_winters_lstm_test.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 计算误差
    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], train_predictions))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], test_predictions))
    trainScore2 = mean_absolute_error(train_data_key[1:], train_predictions)
    testScore2 = mean_absolute_error(test_data_key[1:], test_predictions)
    trainScore3 = r2_score(train_data_key[1:], train_predictions)
    testScore3 = r2_score(test_data_key[1:], test_predictions)
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], train_predictions)
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], test_predictions)

    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})

    df.to_excel(writer, sheet_name='holt_winters_lstm', index=False)

    print(df)
    return test_predictions



def sarima_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 拟合 SARIMA 模型并提取残差
    sarima_model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()
    sarima_train_predictions = sarima_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
    # 训练集预测的第一个值是0
    sarima_train_predictions[0] = train_data_key[0]
    # 计算残差序列
    train_residuals = train_data_key - sarima_train_predictions
    # 归一化残差序列
    scaler = MinMaxScaler()
    scaled_train_residuals = scaler.fit_transform(np.array(train_residuals).reshape(-1, 1))
    # LSTM模型训练和预测
    look_back = 1
    train_X, train_Y = create_sliding_windows(scaled_train_residuals, look_back)
    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)
    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
    # 绘制训练集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key[1:], label='真实值')
    plt.plot(train_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima + lstm: {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_lstm_taian.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # SARIMA模型测试集预测值
    sarima_test_predictions = sarima_model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])
    # 计算残差序列
    sarima_test_residuals = test_data_key - sarima_test_predictions
    # 归一化残差序列
    scaled_test_residuals = scaler.transform(np.array(sarima_test_residuals).reshape(-1, 1))
    # 构造残差数据集
    test_X, test_Y = create_sliding_windows(scaled_test_residuals, look_back)
    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
    test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key[1:], label='真实值')
    plt.plot(test_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima + lstm: {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_lstm_test.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 计算误差
    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], train_predictions))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], test_predictions))
    trainScore2 = mean_absolute_error(train_data_key[1:], train_predictions)
    testScore2 = mean_absolute_error(test_data_key[1:], test_predictions)
    trainScore3 = r2_score(train_data_key[1:], train_predictions)
    testScore3 = r2_score(test_data_key[1:], test_predictions)
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], train_predictions)
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], test_predictions)

    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})

    print(df)

    df.to_excel(writer, sheet_name='sarima_lstm', index=False)

    return test_predictions


def cor_analysis(data):

    # 绘制散点图
    data = pd.DataFrame(data)
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    for i in range(1, 5):
        x = data.iloc[:, i]
        y = data.iloc[:, 0]
        plt.scatter(x, y, color='blue')

        # 进行线性拟合
        slope, intercept = np.polyfit(x, y, 1)
        trendline = intercept + slope * x
        print(trendline)
        # 计算拟合误差
        residuals = y - (slope * x + intercept)
        std_error = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
        trendline_upper = trendline + 1.96 * std_error
        trendline_lower = trendline - 1.96 * std_error

        # 绘制趋势线
        plt.plot(x, trendline, color='red', label='趋势线')
        # 添加误差阴影
        plt.fill_between(x, trendline_upper, trendline_lower, color='red', alpha=0.2, label='Error Range')

        # 添加标签和图例
        plt.xlabel(data.columns[i])
        plt.ylabel(f'{parameters}')
        plt.title(f'{point}')
        plt.legend()
        plt.savefig(f'result/{point}/{data.columns[i]}.jpg', bbox_inches='tight', dpi=600)
        # 显示图形
        plt.show()


def sarima_lstm_v1(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # =================拟合 SARIMA 模型并提取残差==========================
    sarima_model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()
    sarima_train_predictions = sarima_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
    sarima_train_predictions[0] = train_data_key[0]  # 训练集预测的第一个值是0
    # 计算残差序列
    train_residuals = train_data_key - sarima_train_predictions
    # # 归一化残差序列和辅助数据
    mm1 = MinMaxScaler()
    scaled_train_residuals = mm1.fit_transform(np.array(train_residuals).reshape(-1, 1))
    mm2 = MinMaxScaler()
    scaled_train_data_fz = mm2.fit_transform(train_data_fz.values)
    scaled_train_residuals_fz = np.concatenate((scaled_train_residuals, scaled_train_data_fz), axis=1)

    # LSTM模型训练和预测
    look_back = 1
    train_X, train_Y = create_sliding_windows(scaled_train_residuals_fz, look_back)
    # 训练模型  使用ssa找到的最好的神经元个数
    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(look_back, 5)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)
    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = mm1.inverse_transform(lstm_train_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
    # 绘制训练集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key[1:], label='真实值')
    plt.plot(train_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima + lstm(协变量): {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_lstm_v1_taian.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # SARIMA模型测试集预测值
    sarima_test_predictions = sarima_model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])
    # 计算残差序列
    sarima_test_residuals = test_data_key - sarima_test_predictions
    # 归一化残差序列
    scaled_test_residuals = mm1.transform(np.array(sarima_test_residuals).reshape(-1, 1))
    scaled_test_data_fz = mm2.transform(test_data_fz.values)
    scaled_test_residuals_fz = np.concatenate((scaled_test_residuals, scaled_test_data_fz), axis=1)
    # 构造残差数据集
    test_X, test_Y = create_sliding_windows(scaled_test_residuals_fz, look_back)

    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = mm1.inverse_transform(lstm_test_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
    test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key[1:], label='真实值')
    plt.plot(test_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima + lstm(协变量): {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_lstm_v1_test.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 计算误差(预测精度)

    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], train_predictions))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], test_predictions))
    trainScore2 = mean_absolute_error(train_data_key[1:], train_predictions)
    testScore2 = mean_absolute_error(test_data_key[1:], test_predictions)
    trainScore3 = r2_score(train_data_key[1:], train_predictions)
    testScore3 = r2_score(test_data_key[1:], test_predictions)
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], train_predictions)
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], test_predictions)

    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})

    print(df)

    df.to_excel(writer, sheet_name='sarima_lstm_v1', index=False)

    return test_predictions


def sarima_lstm_v2(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # =================拟合 SARIMA 模型并提取残差==========================
    sarima_model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()
    sarima_train_predictions = sarima_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
    sarima_train_predictions[0] = train_data_key[0]  # 训练集预测的第一个值是0
    # 计算残差序列
    train_residuals = train_data_key - sarima_train_predictions
    # # 归一化残差序列和辅助数据
    mm1 = MinMaxScaler()
    scaled_train_residuals = mm1.fit_transform(np.array(train_residuals).reshape(-1, 1))
    mm2 = MinMaxScaler()
    scaled_train_data_fz = mm2.fit_transform(train_data_fz.values)
    scaled_train_residuals_fz = np.concatenate((scaled_train_residuals, scaled_train_data_fz), axis=1)
    # LSTM模型训练和预测
    look_back = 1
    train_X, train_Y = create_sliding_windows(scaled_train_residuals_fz, look_back)

    # ==================================
    UP = [51, 6, 0.055, 9]
    DOWN = [50, 5, 0.05, 8]
    NGEN = 100
    popsize = 100
    parameter = [NGEN, popsize, DOWN, UP]
    # 开始优化
    aco = ACO(parameter)
    aco.main()

    # 训练模型  使用ssa找到的最好的神经元个数
    neurons1 = int(aco.g_best[0])
    neurons2 = int(aco.g_best[1])
    dropout = aco.g_best[2]
    batch_size = int(aco.g_best[3])

    # lstm_model = Sequential()
    # lstm_model.add(LSTM(
    #     input_shape=(look_back, 5),
    #     units=neurons1,
    #     return_sequences=True))
    # lstm_model.add(Dropout(dropout))
    # lstm_model.add(LSTM(
    #     units=neurons2,
    #     return_sequences=False))
    # lstm_model.add(Dropout(dropout))
    # lstm_model.add(Dense(units=1))
    # lstm_model.add(Activation("linear"))
    # lstm_model.compile(loss='mse', optimizer='Adam', metrics='mae')
    # lstm_model.fit(train_X, train_Y, epochs=150, batch_size=batch_size, validation_split=0.2, verbose=1,
    #                callbacks=[EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)])

    inputs = Input(shape=(look_back, 5))

    my_model = LSTM(units=neurons1, activation='tanh', return_sequences=True)(inputs)
    my_model = Dropout(dropout)(my_model)

    my_model = LSTM(units=neurons2, activation='tanh')(my_model)
    my_model = Dropout(dropout)(my_model)

    attention = Dense(units=neurons2, activation='sigmoid', name='attention_vec')(my_model)  # 求解Attention权重
    my_model = Multiply()([my_model, attention])  # attention与LSTM对应数值相乘


    # attention = Dense(units=neurons1, activation='sigmoid', name='attention_vec')(my_model)  # 求解Attention权重
    # my_model = Multiply()([my_model, attention])  # attention与LSTM对应数值相乘

    outputs = Dense(1, activation='tanh')(my_model)
    lstm_model = Model(inputs=inputs, outputs=outputs)
    lstm_model.compile(loss='mse', optimizer='Adam', metrics='mae')
    lstm_model.fit(train_X, train_Y, epochs=150, batch_size=batch_size, validation_split=0.2, verbose=1,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)])
    # =============================================
    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = mm1.inverse_transform(lstm_train_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
    # 绘制训练集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key[1:], label='真实值')
    plt.plot(train_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima + lstm(优化): {point}训练集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_lstm_v2_taian.jpg', bbox_inches='tight', dpi = 600)
    plt.show()
    # SARIMA模型测试集预测值
    sarima_test_predictions = sarima_model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])
    # 计算残差序列
    sarima_test_residuals = test_data_key - sarima_test_predictions
    # 归一化残差序列
    scaled_test_residuals = mm1.transform(np.array(sarima_test_residuals).reshape(-1, 1))
    scaled_test_data_fz = mm2.transform(test_data_fz.values)
    scaled_test_residuals_fz = np.concatenate((scaled_test_residuals, scaled_test_data_fz), axis=1)
    # 构造残差数据集
    test_X, test_Y = create_sliding_windows(scaled_test_residuals_fz, look_back)
    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = mm1.inverse_transform(lstm_test_residuals)
    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
    test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key[1:], label='真实值')
    plt.plot(test_predictions, label='预测值')
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title(f'sarima + lstm(优化): {point}测试集')
    plt.legend()
    plt.savefig(f'result/{point}/sarima_lstm_v2_test.jpg', bbox_inches='tight', dpi = 600)
    plt.show()

    # 计算误差(预测精度)
    trainScore1 = math.sqrt(mean_squared_error(train_data_key[1:], train_predictions))
    testScore1 = math.sqrt(mean_squared_error(test_data_key[1:], test_predictions))
    trainScore2 = mean_absolute_error(train_data_key[1:], train_predictions)
    testScore2 = mean_absolute_error(test_data_key[1:], test_predictions)
    trainScore3 = r2_score(train_data_key[1:], train_predictions)
    testScore3 = r2_score(test_data_key[1:], test_predictions)
    trainScore4 = mean_absolute_percentage_error(train_data_key[1:], train_predictions)
    testScore4 = mean_absolute_percentage_error(test_data_key[1:], test_predictions)

    df = pd.DataFrame({'Train Score: %.2f RMSE': [trainScore1], 'Test Score: %.2f RMSE': [testScore1],
                       'Train Score: %.2f MAE': [trainScore2], 'Test Score: %.2f MAE': [testScore2],
                       'Train Score: %.2f R2': [trainScore3], 'Test Score: %.2f R2': [testScore3],
                       'Train Score: %.2f MAPE': [trainScore4], 'Test Score: %.2f MAPE': [testScore4]})

    print(df)

    df.to_excel(writer, sheet_name='sarima_lstm_v2', index=False)

    return test_predictions


def compare_test_prediction(test_data_key, holt_winters_predictions, sarima_predictions, holt_winters_lstm_predictions, sarima_lstm_predictions, sarima_lstm_v1_predictions, sarima_lstm_v2_predictions):
    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 绘制折线图
    plt.plot(test_data_key.index, test_data_key, label='Actual', marker='+')
    plt.plot(test_data_key.index, holt_winters_predictions, label='holt winters', marker='o')
    plt.plot(test_data_key.index, sarima_predictions, label='sarima', marker='s')
    plt.plot(test_data_key.index[1:], holt_winters_lstm_predictions, label='holt winters+lstm', marker='^')
    plt.plot(test_data_key.index[1:], sarima_lstm_predictions, label='sarima+lstm', marker='*')
    plt.plot(test_data_key.index[1:], sarima_lstm_v1_predictions, label='sarima+lstm(协变量)', marker='d')
    plt.plot(test_data_key.index[1:], sarima_lstm_v2_predictions, label='sarima+lstm(优化)', marker='p')

    # 添加标题和标签
    plt.xlabel('年/月')
    plt.ylabel(f'{parameters}')
    plt.title('Actual vs Predicted')

    # 添加图例
    plt.legend()
    plt.savefig(f'result/{point}/compare_test_prediction.jpg', bbox_inches='tight', dpi = 600)
    plt.show()



if __name__  == '__main__':

    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    for i in range(2, 3):

        point = f'样点{i}'
        parameters = 'TSM'
        data = data_preprocess(rf'data/{point}.xlsx')
        if not os.path.exists(f'result/{point}'):
            os.makedirs(f'result/{point}')
        writer = pd.ExcelWriter(f'result/{point}/{point}.xlsx')

        data_analysis(data[parameters])
        TestStationaryPlot(data[parameters])
        TestStationaryAdfuller(data[parameters])

        train_data_key, train_data_fz, test_data_key, test_data_fz = data_split(data)

        # holt_winters_predictions = holt_winters(train_data_key, train_data_fz, test_data_key, test_data_fz)
        # sarima_predictions = sarima(train_data_key, train_data_fz, test_data_key, test_data_fz)
        # sarima_grid_search(data, parameters)
        #
        # lstm_predictions = lstm(train_data_key, train_data_fz, test_data_key, test_data_fz)
        # holt_winters_lstm_predictions = holt_winters_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz)
        # sarima_lstm_predictions = sarima_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz)
        #
        #
        # cor_analysis(data)
        # sarima_lstm_v1_predictions = sarima_lstm_v1(train_data_key, train_data_fz, test_data_key, test_data_fz)
        sarima_lstm_v2_predictions = sarima_lstm_v2(train_data_key, train_data_fz, test_data_key, test_data_fz)



        # print(len(holt_winters_predictions), len(sarima_predictions), len(holt_winters_lstm_predictions), len(sarima_lstm_predictions))
        # compare_test_prediction(test_data_key, holt_winters_predictions, sarima_predictions, holt_winters_lstm_predictions, sarima_lstm_predictions, sarima_lstm_v1_predictions, sarima_lstm_v2_predictions)

        # writer._save()



