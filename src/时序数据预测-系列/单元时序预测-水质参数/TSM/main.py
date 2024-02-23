import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_percentage_error

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

    for i in cols:
        data[[i]] = replace_outliers(data[[i]])

    for i in cols:
        data[[i]] = fill_missing_data(data[[i]])
    # ===============构造规律的时间间隔========================
    data = data.resample('MS').asfreq()
    data = pd.DataFrame(data)

    return data


def data_analysis(data):
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    # 时序数据分解
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(data)
    result.plot()
    plt.show()

    # ACF：自相关函数
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(data).show()
    plt.show()

    # PACF：偏自相关函数
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(data).show()
    plt.show()

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

    # plt.plot(figsize=(15, 8))
    plt.plot(df, color='blue', label='Original')
    plt.plot(rol_mean, color='red', linestyle='-.', label='Moving Average')
    plt.plot(rol_std, color='black', linestyle='--', label='Standard Deviation')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('TSM', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.title('Moving Average and Standard Deviation', fontsize=14)
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

    dates = data.index
    train_data_key = train_data['TSM']
    train_data_fz = train_data.drop(['TSM'], axis=1)
    test_data_key = test_data['TSM']
    test_data_fz = test_data.drop(['TSM'], axis=1)

    # 绘制训练集和测试集的折线图
    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    plt.plot(train_data_key, label='Training Data')
    plt.plot(test_data_key, label='Testing Data')
    plt.xlabel('Year')
    plt.ylabel('TSM')
    plt.title('TSM-value - Training and Testing Data')
    plt.legend()
    plt.show()

    return train_data_key, train_data_fz, test_data_key, test_data_fz

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)



def holt_winters(train_data_key, train_data_fz, test_data_key, test_data_fz):
    model = ExponentialSmoothing(train_data_key, trend="add", seasonal="add", seasonal_periods=12)
    model_fit = model.fit()
    predictions = model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])# 进行预测

    # 绘制测试集预测结果的折线图
    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key.index, test_data_key, label='Actual')
    plt.plot(predictions.index, predictions, label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('TSM')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    # 绘制原始数据、训练集预测结果和测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(data['TSM'], label='Actual')
    plt.plot(train_data_key.index, model_fit.fittedvalues, label='Training Predictions')
    plt.plot(test_data_key.index, predictions, label='Testing Predictions')
    plt.xlabel('Year')
    plt.ylabel('TSM')
    plt.title('TSM value - Actual vs Predicted')
    plt.legend()
    plt.show()

    # 计算误差
    trainScore = math.sqrt(mean_squared_error(train_data_key[1:], model_fit.fittedvalues[1:]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_data_key[1:], predictions[1:]))
    print('Test Score: %.2f RMSE' % (testScore))

    trainScore = mean_absolute_error(train_data_key[1:], model_fit.fittedvalues[1:])
    print('Train Score: %.2f MAE' % (trainScore))
    testScore = mean_absolute_error(test_data_key[1:], predictions[1:])
    print('Test Score: %.2f MAE' % (testScore))

    trainScore = r2_score(train_data_key[1:], model_fit.fittedvalues[1:])
    print('Train Score: %.2f R2' % (trainScore))
    testScore = r2_score(test_data_key[1:], predictions[1:])
    print('Test Score: %.2f R2' % (testScore))

    trainScore = mean_absolute_percentage_error(train_data_key[1:], model_fit.fittedvalues[1:])
    print('Train Score: %.2f MAPE' % (trainScore))
    testScore = mean_absolute_percentage_error(test_data_key[1:], predictions[1:])
    print('Test Score: %.2f MAPE' % (testScore))

    return predictions


def holt_winters_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 拟合 SARIMA 模型并提取残差
    holt_winters_model = ExponentialSmoothing(train_data_key, trend="add", seasonal="add", seasonal_periods=12)
    holt_winters_model_fit = holt_winters_model.fit()
    holt_winters_train_predictions = holt_winters_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
    # 训练集预测的第一个值是0
    holt_winters_train_predictions[0] = train_data_key[0]
    print(holt_winters_train_predictions, len(holt_winters_train_predictions))

    # 计算残差序列
    train_residuals = train_data_key - holt_winters_train_predictions
    print(train_residuals, len(train_residuals))

    # 归一化残差序列
    scaler = MinMaxScaler()
    scaled_train_residuals = scaler.fit_transform(np.array(train_residuals).reshape(-1, 1))

    # LSTM模型训练和预测
    look_back = 1
    train_X, train_Y = create_dataset(scaled_train_residuals, look_back)

    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)

    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
    print(lstm_train_residuals, len(lstm_train_residuals))

    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = holt_winters_train_predictions[1:] + lstm_train_residuals.flatten()
    print("最终训练集的预测值:", train_predictions)

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
    sarima_test_predictions = holt_winters_model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])
    print(sarima_test_predictions, len(sarima_test_predictions))

    # 计算残差序列
    sarima_test_residuals = test_data_key - sarima_test_predictions

    # 归一化残差序列
    scaled_test_residuals = scaler.transform(np.array(sarima_test_residuals).reshape(-1, 1))

    # 构造残差数据集
    test_X, test_Y = create_dataset(scaled_test_residuals, look_back)

    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
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

    # 计算误差
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

    return test_predictions


def sarima(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 拟合 SARIMA 模型
    model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    # 进行预测
    predictions = model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])


    # 绘制测试集预测结果的折线图
    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_key.index, test_data_key, label='Actual')
    plt.plot(predictions.index, predictions, label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('TSM')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    # 绘制原始数据、训练集预测结果和测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(data['TSM'], label='Actual')
    plt.plot(train_data_key.index, model_fit.fittedvalues, label='Training Predictions')
    plt.plot(test_data_key.index, predictions, label='Testing Predictions')
    plt.xlabel('Year')
    plt.ylabel('TSM')
    plt.title('TSM value - Actual vs Predicted')
    plt.legend()
    plt.show()


    # 计算误差
    trainScore = math.sqrt(mean_squared_error(train_data_key[1:], model_fit.fittedvalues[1:]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_data_key[1:], predictions[1:]))
    print('Test Score: %.2f RMSE' % (testScore))

    trainScore = mean_absolute_error(train_data_key[1:], model_fit.fittedvalues[1:])
    print('Train Score: %.2f MAE' % (trainScore))
    testScore = mean_absolute_error(test_data_key[1:], predictions[1:])
    print('Test Score: %.2f MAE' % (testScore))

    trainScore = r2_score(train_data_key[1:], model_fit.fittedvalues[1:])
    print('Train Score: %.2f R2' % (trainScore))
    testScore = r2_score(test_data_key[1:], predictions[1:])
    print('Test Score: %.2f R2' % (testScore))

    trainScore = mean_absolute_percentage_error(train_data_key[1:], model_fit.fittedvalues[1:])
    print('Train Score: %.2f MAPE' % (trainScore))
    testScore = mean_absolute_percentage_error(test_data_key[1:], predictions[1:])
    print('Test Score: %.2f MAPE' % (testScore))

    return predictions


def sarima_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz):
    # 拟合 SARIMA 模型并提取残差
    sarima_model = SARIMAX(train_data_key, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()
    sarima_train_predictions = sarima_model_fit.predict(start=train_data_key.index[0], end=train_data_key.index[-1])
    # 训练集预测的第一个值是0
    sarima_train_predictions[0] = train_data_key[0]
    print(sarima_train_predictions, len(sarima_train_predictions))

    # 计算残差序列
    train_residuals = train_data_key - sarima_train_predictions
    print(train_residuals, len(train_residuals))

    # 归一化残差序列
    scaler = MinMaxScaler()
    scaled_train_residuals = scaler.fit_transform(np.array(train_residuals).reshape(-1, 1))

    # LSTM模型训练和预测
    look_back = 1
    train_X, train_Y = create_dataset(scaled_train_residuals, look_back)

    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)

    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
    print(lstm_train_residuals, len(lstm_train_residuals))

    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
    print("最终训练集的预测值:", train_predictions)

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
    scaled_test_residuals = scaler.transform(np.array(sarima_test_residuals).reshape(-1, 1))

    # 构造残差数据集
    test_X, test_Y = create_dataset(scaled_test_residuals, look_back)

    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
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

    # 计算误差
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

    return test_predictions


def compare_test_prediction(test_data_key, holt_winters_predictions, sarima_predictions, holt_winters_lstm_predictions, sarima_lstm_predictions):


    # 创建一个新的图形
    plt.figure(figsize=(20, 6))

    # 绘制折线图
    plt.plot(test_data_key.index, test_data_key, label='Actual', marker='+')
    plt.plot(test_data_key.index, holt_winters_predictions, label='holt_winters', marker='o')
    plt.plot(test_data_key.index, sarima_predictions, label='sarima', marker='s')
    plt.plot(test_data_key.index[1:], holt_winters_lstm_predictions, label='holt_winters_lstm', marker='^')
    plt.plot(test_data_key.index[1:], sarima_lstm_predictions, label='sarima_lstm', marker='*')

    # 添加标题和标签
    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.title('Actual vs Predicted')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()



if __name__  == '__main__':

    point = '样点8'
    parameters = 'TSM'
    data = data_preprocess(rf'data/{point}.xlsx')

    data_analysis(data[parameters])

    #对TSM进行平稳性检验
    # TestStationaryPlot(data[parameters])
    # TestStationaryAdfuller(data[parameters])



    train_data_key, train_data_fz, test_data_key, test_data_fz = data_split(data)


    holt_winters_predictions = holt_winters(train_data_key, train_data_fz, test_data_key, test_data_fz)
    sarima_predictions = sarima(train_data_key, train_data_fz, test_data_key, test_data_fz)
    holt_winters_lstm_predictions = holt_winters_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz)
    sarima_lstm_predictions = sarima_lstm(train_data_key, train_data_fz, test_data_key, test_data_fz)
    print(len(holt_winters_predictions), len(sarima_predictions), len(holt_winters_lstm_predictions), len(sarima_lstm_predictions))
    compare_test_prediction(test_data_key, holt_winters_predictions, sarima_predictions, holt_winters_lstm_predictions, sarima_lstm_predictions)



