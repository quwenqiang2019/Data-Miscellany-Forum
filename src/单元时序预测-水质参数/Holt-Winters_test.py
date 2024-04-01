import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


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

# ===============构造规律的时间间隔========================
data = data.resample('MS').asfreq()
data = pd.DataFrame(data)


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
plt.figure(figsize=(10, 6))
plt.plot(train_data_key, label='Training Data')
plt.plot(test_data_key, label='Testing Data')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - Training and Testing Data')
plt.legend()
plt.show()


# 拟合Holt-Winters模型
model = ExponentialSmoothing(train_data_key, trend="add", seasonal="add", seasonal_periods=12)
model_fit = model.fit()
# 进行预测
predictions = model_fit.predict(start=test_data_key.index[0], end=test_data_key.index[-1])


# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_data_key.index, test_data_key, label='Actual')
plt.plot(predictions.index, predictions, label='Predicted')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# 绘制原始数据、训练集预测结果和测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(data['TSM'], label='Actual')
plt.plot(train_data_key.index, model_fit.fittedvalues, label='Training Predictions')
plt.plot(test_data_key.index, predictions, label='Testing Predictions')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - Actual vs Predicted')
plt.legend()
plt.show()