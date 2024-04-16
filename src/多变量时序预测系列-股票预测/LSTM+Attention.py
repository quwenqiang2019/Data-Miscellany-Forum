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

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

window_size = 30
trainX,trainY=createXY(df_for_training_scaled,window_size)
testX,testY=createXY(df_for_testing_scaled,window_size)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

# 创建输入层
# inputs=Input(shape=(window_size, fea_num))
# my_model=LSTM(50, activation='tanh')(inputs)
# attention=Dense(50, activation='sigmoid', name='attention_vec')(my_model)#求解Attention权重
# my_model=Multiply()([my_model, attention])#attention与LSTM对应数值相乘
# outputs = Dense(1, activation='tanh')(my_model)
# my_model = Model(inputs=inputs, outputs=outputs)

inputs = Input(shape=(window_size, fea_num))
lstm = LSTM(50, return_sequences=True)(inputs)
attention = Attention()([lstm, lstm])
attention = Flatten()(attention)
output = Dense(1)(attention)
my_model = Model(inputs=inputs, outputs=output)



my_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
my_model.summary()
my_model.fit(trainX, trainY)

prediction_test=my_model.predict(testX)
print(prediction_test.shape)
prediction_train=my_model.predict(trainX)
prediction_train_copies_array = np.repeat(prediction_train,fea_num, axis=-1)
pred_train=scaler.inverse_transform(np.reshape(prediction_train_copies_array,(len(prediction_train),fea_num)))[:,0]


original_train_copies_array = np.repeat(trainY, fea_num, axis=-1)
original_train=scaler.inverse_transform(np.reshape(original_train_copies_array,(len(trainY),fea_num)))[:,0]
print("train Pred Values-- ", pred_train)
print("\ntrain Original Values-- ", original_train)
plt.plot(df_for_training.index[window_size:,], original_train, color = 'red', label = '真实值')
plt.plot(df_for_training.index[window_size:,], pred_train, color = 'blue', label = '预测值')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Stock Price')
plt.legend()
plt.show()


prediction_test_copies_array = np.repeat(prediction_test,fea_num, axis=-1)
pred_test=scaler.inverse_transform(np.reshape(prediction_test_copies_array,(len(prediction_test),fea_num)))[:,0]
original_test_copies_array = np.repeat(testY, fea_num, axis=-1)
original_test=scaler.inverse_transform(np.reshape(original_test_copies_array,(len(testY),fea_num)))[:,0]
print("test Pred Values-- ", pred_test)
print("\ntest Original Values-- ", original_test)
plt.plot(df_for_testing.index[window_size:,], original_test, color = 'red', label = '真实值')
plt.plot(df_for_testing.index[window_size:,], pred_test, color = 'blue', label = '预测值')
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

