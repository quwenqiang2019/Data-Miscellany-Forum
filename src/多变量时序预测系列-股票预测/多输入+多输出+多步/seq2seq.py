import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, RepeatVector, TimeDistributed
from keras.layers import LSTM
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

# 读取数据
df=pd.read_csv("data.csv", parse_dates=["Date"], index_col=[0])
print(df.shape)
print(df.head())
fea_num = len(df.columns)

# 数据划分
test_split=round(len(df)*0.20)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
# 绘制训练集和测试集的折线图
# 可视化部分
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)

plt.figure(figsize=(10, 6))
plt.plot(df_for_training, label='Training Data')
plt.plot(df_for_testing, label='Testing Data')
plt.xlabel('Day')
plt.ylabel('Open value')
plt.title('Training and Testing Data')
plt.legend()
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
print(len(df_for_training_scaled))


def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations
  #
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
           break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

# 假设给定过去 10 天的观察结果，我们需要预测接下来的 3 天观察结果
n_past = 10
n_future = 3
n_features = fea_num
# # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
X_train, y_train = split_series(df_for_training_scaled,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(df_for_testing_scaled,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

print(X_train)
print(y_train)
print("trainX Shape-- ",X_train.shape)
print("trainY Shape-- ",y_train.shape)
print("testX Shape-- ",X_test.shape)
print("testY Shape-- ",y_test.shape)



# E1D1
# n_features ==> no of features at each timestep in the data.
#
encoder_inputs = Input(shape=(n_past, n_features))
encoder_l1 = LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
decoder_inputs = RepeatVector(n_future)(encoder_outputs1[0])
decoder_l1 = LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = TimeDistributed(Dense(n_features))(decoder_l1)
model_e1d1 = Model(encoder_inputs,decoder_outputs1)
model_e1d1.summary()


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e1d1=model_e1d1.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test),batch_size=32,verbose=0,callbacks=[reduce_lr])



pred_e1d1=model_e1d1.predict(X_test)   # (1029, 3, 5)
print(pred_e1d1)
print(pred_e1d1.shape)


# 以测试集第一个样本进行评价
pred_e1d1_0 = pred_e1d1[0]
print(pred_e1d1_0)
pred_e1d1_0_T = scaler.inverse_transform(pred_e1d1_0)
print(pred_e1d1_0_T)

y_test_0 = y_test[0]
y_test_0_T = scaler.inverse_transform(y_test_0)
print(y_test_0_T)
