import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import tensorflow as tf
import random
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

def read_data(filename):
    # 读取数据集
    data = pd.DataFrame(pd.read_csv(os.path.join(base_dir, 'data', filename)))
    data = data.iloc[0:60,]
    # 将日期列转换为日期时间类型
    data['Month'] = pd.to_datetime(data['Month'])
    # 将日期列设置为索引
    data.set_index('Month', inplace=True)
    return data

def data_split(data):
    # 数据划分
    train_data = data[:train_size]
    test_data = data[train_size:]
    # 绘制训练集和测试集的折线图
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='训练集')
    plt.plot(test_data, label='测试集')
    plt.xlabel('时间序列')
    plt.ylabel('Doseheji')
    plt.title('数据集')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'data_split.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    return train_data, test_data


def df_converse(train_data, test_data):
    train_data_scaler = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_data_scaler = scaler.transform(test_data.values.reshape(-1, 1))

    def createXY(dataset,n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i,0])
        return np.array(dataX),np.array(dataY)

    trainX,trainY=createXY(train_data_scaler,window_size)
    testX,testY=createXY(test_data_scaler,window_size)

    # # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
    testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

    print("trainX Shape-- ",trainX.shape)
    print("trainY Shape-- ",trainY.shape)
    print("testX Shape-- ",testX.shape)
    print("testY Shape-- ",testY.shape)

    return trainX, trainY, testX, testY



def lstm_model():
    # 训练 LSTM 模型
    input = Input(shape=(window_size, 1))
    lstm = LSTM(units=50, activation='relu')(input)
    output = Dense(units=1)(lstm)
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    model.fit(trainX, trainY, epochs=200, batch_size=16)

    # 训练 多层LSTM 模型
    # model = Sequential()
    # model.add(Input(shape=(window_size, fea_num)))
    # model.add(LSTM(50, return_sequences=True))
    # model.add(LSTM(50))
    # model.add(Dense(1))
    # model.summary()
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # model.fit(trainX, trainY, epochs=100, batch_size=32)

    return model


def model_evl():
    # 使用 LSTM 模型进行预测
    train_predictions = model.predict(trainX)
    test_predictions = model.predict(testX)

    # 反归一化预测结果
    pred_train = scaler.inverse_transform(train_predictions)
    pred_test = scaler.inverse_transform(test_predictions)


    # 绘制测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(test_data, label='Actual')
    plt.plot(list(test_data.index)[-len(test_predictions):], pred_test, label='Test Predictions')
    plt.xlabel('Month')
    plt.ylabel('Doseheji')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'test_prediction.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 绘制训练集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Actual')
    plt.plot(list(train_data.index)[window_size:48], pred_train, label='Training Predictions')
    plt.xlabel('Month')
    plt.ylabel('Doseheji')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'train_prediction.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 绘制原始数据、训练集预测结果和测试集预测结果的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Actual')
    plt.plot(list(train_data.index)[window_size:train_size], pred_train, label='Training Predictions')
    plt.plot(list(test_data.index)[-(len(test_data) - window_size):], pred_test, label='Testing Predictions')
    plt.xlabel('Month')
    plt.ylabel('Doseheji')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'train_test_prediction.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


    # 计算误差
    original_test = test_data.iloc[window_size:,]
    original_train = train_data.iloc[window_size:,]
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

    index = ['Test Score: %.2f RMSE',
              'Test Score: %.2f MAE',
              'Test Score: %.2f R2',
              'Test Score: %.2f MAPE',
              'Train Score: %.2f RMSE',
              'Train Score: %.2f MAE',
              'Train Score: %.2f R2',
              'Train Score: %.2f MAPE']

    values = [testScore1,  testScore2, testScore3, testScore4, trainScore1, trainScore2, trainScore3, trainScore4]
    df = pd.DataFrame({'index': index, 'values': values})
    print(df)

    df.to_excel(os.path.join(base_dir, 'result', 'eval.xlsx'), index=False)   # 将评估指标值存为数据表

    return


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    fea_num = 1
    window_size = 1
    train_size = 48
    scaler = MinMaxScaler()

    filename = 'Doseheji.csv'
    data = read_data(filename)
    train_data, test_data = data_split(data)
    trainX, trainY, testX, testY = df_converse(train_data, test_data)
    model = lstm_model()
    model_evl()

