import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import itertools
import statsmodels.api as sm


# 读取数据集
data = pd.read_excel('样点5.xlsx')
# data = pd.read_excel('5样点.xlsx', sheet_name='Sheet2')
# data = data[['日期', 'TSM']]
data = pd.DataFrame(data)
data['日期'] = pd.to_datetime(data['日期'], format='%Y%m')# 将日期列转换为日期时间类型
data['TSM'] = data['TSM'].interpolate()   # 使用插值法填充缺失值
data.set_index('日期', inplace=True) # 将日期列设置为索引
data = data.resample('MS').asfreq() # 构造规律的时间间隔
data['TSM'] = data['TSM'].interpolate() # 使用插值法填充缺失值



#====================异常值处理========================
def replace_outliers(series):
    # 计算均值和标准差
    mean = series.mean().values[0]
    std = series.std().values[0]
    # 设置阈值
    threshold = mean + 2 * std
    print(threshold)

    for date in series.index:
        year = date.year
        month = date.month

        if series.loc[date].values[0] > threshold:
            same_month_data = series[(series.index.year != year) & (series.index.month == month)]
            month_mean = same_month_data.mean()
            series.loc[date] = month_mean
    return series

data = replace_outliers(data)

print(data.head(10))


#====================缺失值处理========================
def fill_missing_data(series):
    for date in series.index:
        year = date.year
        month = date.month

        if pd.isnull(series.loc[date].values[0]):
            print(series.loc[date].values[0])
            same_month_data = series[(series.index.year != year) & (series.index.month == month)]
            month_mean = same_month_data.mean()
            series.loc[date] = month_mean
    return series

data = fill_missing_data(data)
print(data)



# 检验时间序列平稳性
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
    plt.ylabel('CO2', fontsize=16)
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


#对CO2进行平稳性检验

TestStationaryPlot(data)
TestStationaryAdfuller(data)



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



warnings.filterwarnings('ignore')

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


#模型的验证
results.plot_diagnostics(figsize=(12, 10))
plt.show()


# 拆分数据集为训练集和测试集
train_data = data.iloc[:-15]
print(train_data)
test_data = data.iloc[-15:]
print(test_data)

# 绘制训练集和测试集的折线图
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - Training and Testing Data')
plt.legend()
plt.show()


# 拟合 SARIMA 模型
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()
# 进行预测
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])


# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(predictions.index, predictions, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# 绘制原始数据、训练集预测结果和测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(data, label='Actual')
plt.plot(train_data.index, model_fit.fittedvalues, label='Training Predictions')
plt.plot(test_data.index, predictions, label='Testing Predictions')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - Actual vs Predicted')
plt.legend()
plt.show()


# 计算误差
trainScore = math.sqrt(mean_squared_error(train_data, model_fit.fittedvalues))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_data, predictions))
print('Test Score: %.2f RMSE' % (testScore))