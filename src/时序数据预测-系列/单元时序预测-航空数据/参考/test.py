import pandas as pd

# 读取数据集
data = pd.read_csv('international-airline-passengers.csv')

# 将日期列转换为日期时间格式
data['Month'] = pd.to_datetime(data['Month'])

# 将日期列设置为索引
data.set_index('Month', inplace=True)

# 可选：绘制数据集的折线图，查看数据的趋势和季节性
import matplotlib.pyplot as plt
plt.plot(data)
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers')
plt.show()


# 拆分数据集为训练集和测试集
train_data = data[:'1959']
test_data = data['1960':]

from statsmodels.tsa.statespace.sarimax import SARIMAX

# 选择模型参数
order = (1, 1, 1)  # (AR, I, MA)
seasonal_order = (1, 1, 1, 12)  # (AR, I, MA, 季节性周期)

# 拟合SARIMA模型
model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
model_fit = model.fit()

# 预测训练集中的值
train_predictions = model_fit.predict(start=train_data.index[0], end=train_data.index[-1])


# 预测测试集中的值
test_predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

# 可选：绘制训练集和测试集的预测结果
plt.plot(train_data, label='Train')
plt.plot(test_data, label='Test')
plt.plot(train_predictions, label='Train Predictions')
plt.plot(test_predictions, label='Test Predictions')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('International Airline Passengers - SARIMA Predictions')
plt.legend()
plt.show()