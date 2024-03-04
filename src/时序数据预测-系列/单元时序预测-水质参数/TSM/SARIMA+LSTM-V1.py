import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import math

import itertools
import statsmodels.api
import warnings
warnings.filterwarnings('ignore')


# =============================读取数据集===============================
data = pd.read_excel(r'data/样点8.xlsx')
data = pd.DataFrame(data)
# 将日期列转换为日期时间类型
data['日期'] = pd.to_datetime(data['日期'], format='%Y%m')
# 将日期列设置为索引
data.set_index('日期', inplace=True)
cols = list(data.columns)


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

for i in cols:
    data[[i]] = replace_outliers(data[[i]])


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


for i in cols:
    data[[i]] = fill_missing_data(data[[i]])

# ===========================================构造规律的时间间隔=============================================
data = data.resample('MS').asfreq()
data = pd.DataFrame(data)


# ============================================检验时间序列平稳性===========================================
# 方法一：绘制移动平均值和标准差
def TestStationaryPlot(df):
    rol_mean = df.rolling(window=12, center=False).mean()
    rol_std = df.rolling(window=12, center=False).std()

    # plt.plot(figsize=(15, 8))
    plt.plot(df, color='blue', label='Original')
    plt.plot(rol_mean, color='red', linestyle='-.', label='Moving Average')
    plt.plot(rol_std, color='black', linestyle='--', label='Standard Deviation')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel('Time', fontsize=16)
    plt.ylabel('TSM', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.title('Moving Average and Standard Deviation', fontsize=22)
    plt.show(block=True)


#方法二：ADF检验
from statsmodels.tsa.stattools import adfuller
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


#对TSM进行平稳性检验
TestStationaryPlot(data['TSM'])
TestStationaryAdfuller(data['TSM'])

# ========================== 划分数据集 ==================================
train_size = len(data) - 15
train_data = data[:train_size]
test_data = data[train_size:]

dates = data.index
train_data_key = train_data['TSM']
train_data_fz = train_data.drop(['TSM'], axis=1)
test_data_key = test_data['TSM']
test_data_fz = test_data.drop(['TSM'], axis=1)

# =================拟合 SARIMA 模型并提取残差==========================
sarima_model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()
# sarima_train_predictions = sarima_model_fit.predict(start=0, end=train_size-1)
sarima_train_predictions = sarima_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
sarima_train_predictions[0] = train_data_key[0]      # 训练集预测的第一个值是0
print(sarima_train_predictions, len(sarima_train_predictions))

# 计算残差序列
train_residuals = train_data_key - sarima_train_predictions
print(train_residuals, len(train_residuals))

# # 归一化残差序列和辅助数据
mm1 = MinMaxScaler()
scaled_train_residuals = mm1.fit_transform(np.array(train_residuals).reshape(-1, 1))
mm2 = MinMaxScaler()
scaled_train_data_fz = mm2.fit_transform(train_data_fz.values)
scaled_train_residuals_fz = np.concatenate((scaled_train_residuals, scaled_train_data_fz), axis=1)
print(scaled_train_residuals_fz, scaled_train_residuals_fz.shape)
#
#
# LSTM模型训练和预测
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 1
train_X, train_Y = create_dataset(scaled_train_residuals_fz, look_back)

# 训练模型  使用ssa找到的最好的神经元个数
lstm_model = Sequential()
lstm_model.add(LSTM(4, input_shape=(look_back, 5)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)

# LSTM模型预测整个训练集的残差值
lstm_train_residuals = lstm_model.predict(train_X)
lstm_train_residuals = mm1.inverse_transform(lstm_train_residuals)
print(lstm_train_residuals, len(lstm_train_residuals))
#
# SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
print("最终训练集的预测值:", train_predictions)
#
# 绘制训练集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(train_predictions, label='Predicted')
plt.plot(train_data_key[1:], label='Actual')
plt.xlabel('Month')
plt.ylabel('TSM')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()


# SARIMA模型测试集预测值
sarima_test_predictions = sarima_model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])
print(sarima_test_predictions, len(sarima_test_predictions))

# 计算残差序列
sarima_test_residuals = test_data_key - sarima_test_predictions

# 归一化残差序列
scaled_test_residuals = mm1.transform(np.array(sarima_test_residuals).reshape(-1, 1))
scaled_test_data_fz = mm2.transform(test_data_fz.values)

scaled_test_residuals_fz = np.concatenate((scaled_test_residuals, scaled_test_data_fz), axis=1)
print(scaled_test_residuals_fz, scaled_test_residuals_fz.shape)




# 构造残差数据集
test_X, test_Y = create_dataset(scaled_test_residuals_fz, look_back)

# LSTM模型预测整个测试集的残差值
lstm_test_residuals = lstm_model.predict(test_X)
lstm_test_residuals = mm1.inverse_transform(lstm_test_residuals)
print(lstm_test_residuals, len(lstm_test_residuals))

# SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
print("最终测试集的预测值:", test_predictions)

# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_predictions, label='Predicted')
plt.plot(test_data_key[1:], label='Actual')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()


# 计算误差(预测精度)


trainScore = math.sqrt(mean_squared_error(train_data_key[1:], train_predictions))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_data_key[1:], test_predictions))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_error(train_data_key[1:], train_predictions)
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(test_data_key[1:], test_predictions)
print('Test Score: %.2f MAE' % (testScore))

trainScore = r2_score(train_data_key[1:], train_predictions)
print('Train Score: %.2f R2' % (trainScore))
testScore = r2_score(test_data_key[1:], test_predictions)
print('Test Score: %.2f R2' % (testScore))

trainScore = mean_absolute_percentage_error(train_data_key[1:], train_predictions)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(test_data_key[1:], test_predictions)
print('Test Score: %.2f MAPE' % (testScore))