import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# 读取数据集
data = pd.read_csv('data.csv')
# 将日期列转换为日期时间类型
data['Month'] = pd.to_datetime(data['Month'])
# 将日期列设置为索引
data.set_index('Month', inplace=True)


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


# 拟合ARMA模型
model = ARIMA(train_data, order=(2, 0, 2))
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