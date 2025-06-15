import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


# 读取数据集
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', 'data1.xlsx')))
data = df[['month', 'flu_rate']]
# 将日期列转换为日期时间类型
data['month'] = pd.to_datetime(data['month'])
# 将日期列设置为索引
data.set_index('month', inplace=True)


# 拆分数据集为训练集和测试集
train_size = len(data) - 12
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# 绘制训练集和测试集的折线图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'SARIMA1.tif'))
plt.show()


# 拟合 SARIMA 模型
model = SARIMAX(train_data, exog=df[['temperature', 'rain']].values[:train_size],order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()
# 进行预测
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1], exog=df[['temperature', 'rain']].values[train_size:])

# 绘制测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(predictions.index, predictions, label='Predicted')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.title('Actual vs Predicted')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'SARIMA2.tif'))
plt.show()

# 绘制原始数据、训练集预测结果和测试集预测结果的折线图
plt.figure(figsize=(10, 6))
plt.plot(data, label='Actual')
plt.plot(train_data.index, model_fit.fittedvalues, label='Training Predictions')
plt.plot(test_data.index, predictions, label='Testing Predictions')
plt.xlabel('month')
plt.ylabel('flu_rate')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'SARIMA3.tif'))
plt.show()