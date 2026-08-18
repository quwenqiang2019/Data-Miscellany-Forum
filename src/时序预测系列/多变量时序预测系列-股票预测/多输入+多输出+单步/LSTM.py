import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention, Multiply, Flatten
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
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

def createXY(data, win_size, target_feature_idxs):
    dataX = []
    dataY = []
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size, :]
        temp_y = [data[i + win_size, idx] for idx in target_feature_idxs]
        dataX.append(temp_x)
        dataY.append(temp_y)
    return np.array(dataX),np.array(dataY)

win_size = 12 # 时间窗口
target_feature_idxs = [0, 1, 2, 3, 4] # 指定待预测特征列索引
trainX, trainY = createXY(df_for_training_scaled, win_size, target_feature_idxs)
testX, testY = createXY(df_for_testing_scaled, win_size, target_feature_idxs)
print("训练集形状:", trainX.shape, trainY.shape)
print("测试集形状:", testX.shape, testY.shape)
# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], win_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], win_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

# 输入维度
input_shape = Input(shape=(trainX.shape[1], trainX.shape[2]))
# LSTM层
lstm_layer = LSTM(128, activation='relu')(input_shape)
# 全连接层
dense_1 = Dense(64, activation='relu')(lstm_layer)
dense_2 = Dense(32, activation='relu')(dense_1)
# 输出层
output_1 = Dense(1, name='Open')(dense_2)
output_2 = Dense(1, name='High')(dense_2)
output_3 = Dense(1, name='Low')(dense_2)
output_4 = Dense(1, name='Close')(dense_2)
output_5 = Dense(1, name='AdjClose')(dense_2)
model = Model(inputs = input_shape, outputs = [output_1, output_2, output_3, output_4, output_5])
model.compile(loss='mse', optimizer='adam')
model.summary()
# # 模型拟合
history = model.fit(trainX, [trainY[:,i] for i in range(trainY.shape[1])], epochs=10, batch_size=32)
plt.figure()
plt.plot(history.history['loss'], c='b', label='loss')
plt.legend()
plt.show()


# def predict_next_11_days(model, input_data):
#     input_sequence = input_data.copy()
#
#     # 预测未来 11 天的数据
#     future_predictions = []
#     for _ in range(11):
#         predictions = model.predict(np.expand_dims(input_sequence[-1], axis=0))
#         next_data = np.append(input_sequence[-1, 1:], np.array(predictions).reshape(1, 5), axis=0)
#         input_sequence = np.append(input_sequence, [next_data], axis=0)
#         future_predictions.append(predictions)
#     future_predictions = np.array(future_predictions).reshape(11, 5)
#
#     return future_predictions
#
#
# future_predictions = predict_next_11_days(model, testX[-1:])
# print(future_predictions)


i=0
prediction_train = model.predict(trainX)
prediction_train0=model.predict(trainX)[i]
prediction_train_copies_array = np.repeat(prediction_train0,fea_num, axis=-1)
pred_train=scaler.inverse_transform(np.reshape(prediction_train_copies_array,(len(prediction_train0),fea_num)))[:,i]
original_train_copies_array = trainY
original_train=scaler.inverse_transform(np.reshape(original_train_copies_array,(len(trainY),fea_num)))[:,i]
print("train Pred Values-- ", pred_train)
print("\ntrain Original Values-- ", original_train)
plt.plot(df_for_training.index[win_size:,], original_train, color = 'red', label = '真实值')
plt.plot(df_for_training.index[win_size:,], pred_train, color = 'blue', label = '预测值')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Stock Price')
plt.legend()
plt.show()

prediction_test = model.predict(testX)
prediction_test0=model.predict(testX)[i]
prediction_test_copies_array = np.repeat(prediction_test0,fea_num, axis=-1)
pred_test=scaler.inverse_transform(np.reshape(prediction_test_copies_array,(len(prediction_test0),fea_num)))[:,i]
original_test_copies_array = testY
original_test=scaler.inverse_transform(np.reshape(original_test_copies_array,(len(testY),fea_num)))[:,i]
print("test Pred Values-- ", pred_test)
print("\ntest Original Values-- ", original_test)
plt.plot(df_for_testing.index[win_size:,], original_test, color = 'red', label = '真实值')
plt.plot(df_for_testing.index[win_size:,], pred_test, color = 'blue', label = '预测值')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Stock Price')
plt.legend()
plt.show()




# 计算误差
testScore1 = math.sqrt(mean_squared_error(original_test, pred_test))
print('Test Score: %.2f RMSE' % (testScore1))
testScore2 = mean_absolute_error(original_test, pred_test)
print('Test Score: %.2f MAE' % (testScore2))
testScore3 = r2_score(original_test, pred_test)
print('Test Score: %.2f R2' % (testScore3))
testScore4 = mean_absolute_percentage_error(original_test, pred_test)
print('Test Score: %.2f MAPE' % (testScore4))

trainScore1 = math.sqrt(mean_squared_error(original_train, pred_train))
print('train Score: %.2f RMSE' % (trainScore1))
trainScore2 = mean_absolute_error(original_train, pred_train)
print('train Score: %.2f MAE' % (trainScore2))
trainScore3 = r2_score(original_train, pred_train)
print('train Score: %.2f R2' % (trainScore3))
trainScore4 = mean_absolute_percentage_error(original_train, pred_train)
print('train Score: %.2f MAPE' % (trainScore4))

df = pd.DataFrame({'Test Score: %.2f RMSE': [testScore1],
                   'Test Score: %.2f MAE': [testScore2],
                   'Test Score: %.2f R2': [testScore3],
                   'Test Score: %.2f MAPE': [testScore4],
                   'Train Score: %.2f RMSE': [trainScore1],
                   'Train Score: %.2f MAE': [trainScore2],
                   'Train Score: %.2f R2': [trainScore3],
                   'Train Score: %.2f MAPE': [trainScore4]
                   })

