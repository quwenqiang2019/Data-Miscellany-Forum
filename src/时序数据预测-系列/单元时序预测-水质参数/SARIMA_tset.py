import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
# 读取数据集
# data = pd.read_excel('样点5.xlsx')
data = pd.read_excel('5样点 - 副本 - 副本.xlsx')
data = pd.DataFrame(data)

data = data[['日期', 'TSMvalue']]

# 将日期列转换为日期时间类型
data['日期'] = pd.to_datetime(data['日期'], format='%Y%m')

# # 使用插值法填充缺失值
# data['TSM值'] = data['TSM值'].interpolate()
# 将日期列设置为索引
data.set_index('日期', inplace=True)
# # 构造规律的时间间隔
# data = data.resample('MS').asfreq()
# # 使用插值法填充缺失值
# data['TSM值'] = data['TSM值'].interpolate()
# print(data.index)

#====================异常值处理========================
def replace_outliers(series):
    # 计算均值和标准差
    mean = series.mean().values[0]
    std = series.std().values[0]
    # 设置阈值
    threshold = mean + 2 * std

    threshold = 150

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

# 拆分数据集为训练集和测试集
train_data = data.iloc[:-12]
test_data = data.iloc[-12:]

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
# predictions = model_fit.forecasts(len(test_data))

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