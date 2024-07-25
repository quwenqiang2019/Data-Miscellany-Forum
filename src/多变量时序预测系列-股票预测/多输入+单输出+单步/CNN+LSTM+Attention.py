import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from keras.layers import *
from keras.models import *
from keras.layers.merging.concatenate import concatenate

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

window_size = 3
trainX,trainY=createXY(df_for_training_scaled,window_size)
testX,testY=createXY(df_for_testing_scaled,window_size)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)


# 注意力机制
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    # 根据给定的模式(dim)置换输入的维度  例如(2,1)即置换输入的第1和第2个维度
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # Layer that multiplies (element-wise) a list of inputs.
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# ------------------------------------------------------------------------------------------------------#
#   注意力模块，主要是实现对step维度的注意力机制
#   在这里大家可能会疑惑，为什么需要先Permute再进行注意力机制的施加。
#   这是因为，如果我们直接进行全连接的话，我们的最后一维是特征维度，这个时候，我们每个step的特征是分开的，
#   此时进行全连接的话，得出来注意力权值每一个step之间是不存在特征交换的，自然也就不准确了。
#   所以在这里我们需要首先将step维度转到最后一维，然后再进行全连接，根据每一个step的特征获得注意力机制的权值。
def attention_block(inputs, time_step):
    # batch_size, time_steps, lstm_units -> batch_size, lstm_units, time_steps
    a = Permute((2, 1))(inputs)
    # batch_size, lstm_units, time_steps -> batch_size, lstm_units, time_steps
    a = Dense(time_step, activation='softmax')(a)  # 和步长有关
    # batch_size, lstm_units, time_steps -> batch_size, time_steps, lstm_units
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # 相当于获得每一个step中，每个特征的权重
    output_attention_mul = concatenate([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# 建立cnn-LSTM-并添加注意力机制
inputs = Input(shape=(window_size, fea_num))
x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
x = Dropout(0.3)(x)
lstm_out = CuDNNLSTM(50, return_sequences=True)(x)
lstm_out = Dropout(0.3)(lstm_out)
# attention_mul = attention_3d_block(lstm_out)
attention_mul = attention_block(lstm_out, window_size)
attention_mul = Flatten()(attention_mul)
output = Dense(1, activation='linear')(attention_mul)
my_model = Model(inputs=[inputs], outputs=output)

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

