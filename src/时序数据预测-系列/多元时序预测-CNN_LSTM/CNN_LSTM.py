import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from numpy.random import seed

#加载数据
train = pd.read_csv('CNN-LSTM/train.csv', parse_dates=['date'])
test = pd.read_csv('CNN-LSTM/test.csv', parse_dates=['date'])


# 训练数据集的时间周期
print('Min date from train set: %s' % train['date'].min().date())
print('Max date from train set: %s' % train['date'].max().date())

lag_size = (test['date'].max().date() - train['date'].max().date()).days
print('Max date from train set: %s' % train['date'].max().date())
print('Max date from test set: %s' % test['date'].max().date())
print('Forecast lag size', lag_size)

daily_sales = train.groupby('date', as_index=False)['sales'].sum()
store_daily_sales = train.groupby(['store', 'date'], as_index=False)['sales'].sum()
item_daily_sales = train.groupby(['item', 'date'], as_index=False)['sales'].sum()

import seaborn as sns
import matplotlib.pyplot as plt

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("bright")  # 设置亮色调色板

# 创建图像
fig, ax = plt.subplots(figsize=(10, 6))  # 设置图像大小

# 绘制折线图
sns.lineplot(data=daily_sales, x='date', y='sales', ax=ax)

# 设置标题和轴标签
ax.set_title('Daily Sales', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)

# 调整刻度标签字体大小
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# 调整图例样式和位置
ax.legend(['Sales'], loc='upper left', fontsize=10)

# 保存图像（可选）
plt.savefig('daily_sales_plot.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()


# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("bright")  # 设置亮色调色板

# 创建图像
fig, ax = plt.subplots(figsize=(10, 6))  # 设置图像大小

# 遍历每个店铺的销售数据
for store in store_daily_sales['store'].unique():
    current_store_daily_sales = store_daily_sales[store_daily_sales['store'] == store]
    
    # 绘制折线图
    sns.lineplot(data=current_store_daily_sales, x='date', y='sales', label=f"Store {store}", ax=ax)

# 设置标题和轴标签
ax.set_title('Store Daily Sales', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)

# 调整刻度标签字体大小
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# 调整图例样式和位置
ax.legend(loc='upper left', fontsize=10)

# 保存图像（可选）
plt.savefig('store_daily_sales_plot.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()


# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("bright")  # 设置亮色调色板

# 创建图像
fig, ax = plt.subplots(figsize=(10, 6))  # 设置图像大小

# 遍历每个商品的销售数据
max_legend_items = 5  # 最多显示的图例项数
num_legend_items = 0  # 当前已显示的图例项数

for item in item_daily_sales['item'].unique():
    current_item_daily_sales = item_daily_sales[item_daily_sales['item'] == item]
    
    # 只绘制前 max_legend_items 个图例项
    if num_legend_items < max_legend_items:
        # 绘制折线图
        sns.lineplot(data=current_item_daily_sales, x='date', y='sales', label=f"Item {item}", ax=ax)
        num_legend_items += 1
    else:
        break

# 添加省略号标识
if num_legend_items < len(item_daily_sales['item'].unique()):
    ax.plot([], [], ' ', label='...')

# 设置标题和轴标签
ax.set_title('Item Daily Sales', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)

# 调整刻度标签字体大小
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# 调整图例样式和位置
ax.legend(loc='upper left', fontsize=10)

# 保存图像（可选）
plt.savefig('item_daily_sales_plot.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()



train = train[(train['date'] >= '2017-01-01')]
train_gp = train.sort_values('date').groupby(['item', 'store', 'date'], as_index=False)
train_gp = train_gp.agg({'sales':['mean']})
train_gp.columns = ['item', 'store', 'date', 'sales']

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

window = 29
lag = lag_size
series = series_to_supervised(train_gp.drop('date', axis=1), window=window, lag=lag)
series.head()

last_item = 'item(t-%d)' % window
last_store = 'store(t-%d)' % window
series = series[(series['store(t)'] == series[last_store])]
series = series[(series['item(t)'] == series[last_item])]



columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item', 'store']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item', 'store']]
series.drop(columns_to_drop, axis=1, inplace=True)
series.drop(['item(t)', 'store(t)'], axis=1, inplace=True)



# Label
labels_col = 'sales(t+%d)' % lag_size
labels = series[labels_col]
series = series.drop(labels_col, axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
X_train.head()

X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)


subsequences = 2
timesteps = X_train_series.shape[1]//subsequences
X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))
print('Train set shape', X_train_series_sub.shape)
print('Validation set shape', X_valid_series_sub.shape)



model_cnn_lstm = Sequential()
model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model_cnn_lstm.add(TimeDistributed(Flatten()))
model_cnn_lstm.add(LSTM(50, activation='relu'))
model_cnn_lstm.add(Dense(1))
optimizer = tf.keras.optimizers.Adam() 
print("OK")

model_cnn_lstm.compile(loss='mse', optimizer=optimizer)


cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=50, verbose=2)

# 设置样式和配色方案
sns.set_style("whitegrid")
sns.set_palette("bright")

# 创建一个子图对象
fig, ax4 = plt.subplots()

# 绘制训练损失和验证损失
ax4.plot(cnn_lstm_history.history['loss'], label='Train loss')
ax4.plot(cnn_lstm_history.history['val_loss'], label='Validation loss')
ax4.legend(loc='best')
ax4.set_title('CNN-LSTM')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('MSE')

# 设置坐标轴标签的字体大小
ax4.xaxis.label.set_size(12)
ax4.yaxis.label.set_size(12)

# 设置图例的字体大小
ax4.legend(loc='best', fontsize='medium')

# 设置标题的字体大小
ax4.set_title('CNN-LSTM', fontsize=16)

# 增加网格线的透明度
ax4.grid(alpha=0.5)

# 显示图形
plt.tight_layout()
plt.show()



cnn_lstm_train_pred = model_cnn_lstm.predict(X_train_series_sub)
cnn_lstm_valid_pred = model_cnn_lstm.predict(X_valid_series_sub)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, cnn_lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, cnn_lstm_valid_pred)))





