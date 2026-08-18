import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import *
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from keras.layers import CuDNNLSTM
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

def read_data(filename):
    # 读取数据集
    df = pd.DataFrame(pd.read_csv(os.path.join(base_dir, 'data', filename), encoding='ANSI'))
    # 将日期列设置为索引
    df.set_index('时间', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)

    return df


def lstm_model(df):
    fea_num = len(df.columns)
    # 数据划分
    test_split=round(len(df)*0.20)
    # test_split = len(df) - 12
    print(test_split)
    df_for_training=df[:-test_split]
    df_for_testing=df[-test_split:]
    # 绘制训练集和测试集的折线图
    # 可视化部分
    sns.set(font_scale=1.2)
    plt.rc('font', family=['SimSun'], size=12)
    # plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(10, 6))
    plt.plot(df_for_training[target], label='训练集')
    plt.plot(df_for_testing[target], label='测试集')
    plt.xlabel('时间序列')
    plt.ylabel(target)
    plt.title('数据集')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'lstm_data_split.jpg'), bbox_inches='tight', dpi=600)
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

    window_size = 2
    trainX,trainY=createXY(df_for_training_scaled,window_size)
    testX,testY=createXY(df_for_testing_scaled,window_size)

    # # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
    testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

    print("trainX Shape-- ",trainX.shape)
    print("trainY Shape-- ",trainY.shape)
    print("testX Shape-- ",testX.shape)
    print("testY Shape-- ",testY.shape)

    my_model = Sequential()
    my_model.add(Input(shape=(window_size, fea_num)))
    my_model.add(LSTM(10, return_sequences=True))
    my_model.add(LSTM(8))
    my_model.add(Dense(1))

    my_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    my_model.summary()
    my_model.fit(trainX, trainY)

    prediction_test=my_model.predict(testX)
    prediction_train=my_model.predict(trainX)

    prediction_train_copies_array = np.repeat(prediction_train,fea_num, axis=-1)
    pred_train=scaler.inverse_transform(np.reshape(prediction_train_copies_array,(len(prediction_train),fea_num)))[:,0]
    original_train_copies_array = np.repeat(trainY, fea_num, axis=-1)
    original_train=scaler.inverse_transform(np.reshape(original_train_copies_array,(len(trainY),fea_num)))[:,0]
    print("train Pred Values-- ", pred_train)
    print("\ntrain Original Values-- ", original_train)

    plt.plot(df_for_training.index[window_size:,], original_train, color = 'red', label = '真实值')
    plt.plot(df_for_training.index[window_size:,], pred_train, color = 'blue', label = '预测值')
    plt.title(f'{target}预测')
    plt.xlabel('时间序列')
    plt.xticks(rotation=45)
    plt.ylabel(target)
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'lstm_train_pred.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


    prediction_test_copies_array = np.repeat(prediction_test,fea_num, axis=-1)
    pred_test=scaler.inverse_transform(np.reshape(prediction_test_copies_array,(len(prediction_test),fea_num)))[:,0]
    original_test_copies_array = np.repeat(testY, fea_num, axis=-1)
    original_test=scaler.inverse_transform(np.reshape(original_test_copies_array,(len(testY),fea_num)))[:,0]
    print("test Pred Values-- ", pred_test)
    print("\ntest Original Values-- ", original_test)

    plt.plot(df_for_testing.index[window_size:,], original_test, color = 'red', label = '真实值')
    plt.plot(df_for_testing.index[window_size:,], pred_test, color = 'blue', label = '预测值')
    plt.title(f'{target}预测')
    plt.xlabel('时间序列')
    plt.xticks(rotation=45)
    plt.ylabel(target)
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'lstm_test_pred.jpg'), bbox_inches='tight', dpi=600)
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

    df.to_excel(os.path.join(base_dir, 'result', 'lstm_eval.xlsx'), index=False)   # 将评估指标值存为数据表



def sarima_lstm(df):
    df = df[target].values
    # 拆分数据集为训练集和测试集
    train_size = int(len(df) * 0.8)
    # train_size = len(data) - 12
    train_data = df[:train_size]
    test_data = df[train_size:]
    print(train_data, len(train_data))

    # 拟合 SARIMA 模型并提取残差
    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    sarima_model_fit = sarima_model.fit()
    sarima_train_predictions = sarima_model_fit.predict(start=0, end=train_size - 1)
    # 训练集预测的第一个值是0
    sarima_train_predictions[0] = train_data[0]
    print(sarima_train_predictions, len(sarima_train_predictions))

    # 计算残差序列
    train_residuals = train_data - sarima_train_predictions
    print(train_residuals, len(train_residuals))

    # 归一化残差序列
    scaler = MinMaxScaler()
    scaled_train_residuals = scaler.fit_transform(train_residuals.reshape(-1, 1))

    # LSTM模型训练和预测
    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            Y.append(data[i + look_back])
        return np.array(X), np.array(Y)

    look_back = 1
    train_X, train_Y = create_dataset(scaled_train_residuals, look_back)

    lstm_model = Sequential()
    lstm_model.add(LSTM(4, input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)

    # LSTM模型预测整个训练集的残差值
    lstm_train_residuals = lstm_model.predict(train_X)
    lstm_train_residuals = scaler.inverse_transform(lstm_train_residuals)
    print(lstm_train_residuals, len(lstm_train_residuals))  # look_back = 1，第一个残差无法预测

    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终训练集的预测值
    train_predictions = sarima_train_predictions[1:] + lstm_train_residuals.flatten()
    print("最终训练集的预测值:", train_predictions)

    # 绘制训练集预测结果的折线图
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(10, 6))
    plt.plot(train_predictions, label='Predicted')
    plt.plot(train_data[1:], label='Actual')
    plt.xlabel('Sequences')
    plt.ylabel(target)
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'sarima_lstm_train_pred.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # SARIMA模型测试集预测值
    sarima_test_predictions = sarima_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    print(sarima_test_predictions, len(sarima_test_predictions))

    # 计算残差序列
    sarima_test_residuals = test_data - sarima_test_predictions

    # 归一化残差序列
    scaled_test_residuals = scaler.transform(sarima_test_residuals.reshape(-1, 1))

    # 构造残差数据集
    test_X, test_Y = create_dataset(scaled_test_residuals, look_back)

    # LSTM模型预测整个测试集的残差值
    lstm_test_residuals = lstm_model.predict(test_X)
    lstm_test_residuals = scaler.inverse_transform(lstm_test_residuals)
    print(lstm_test_residuals, len(lstm_test_residuals))

    # SARIMA模型预测值与LSTM模型预测残差值相加得到最终测试集的预测值
    test_predictions = sarima_test_predictions[1:] + lstm_test_residuals.flatten()
    print("最终测试集的预测值:", test_predictions)
    # 绘制测试集预测结果的折线图
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(10, 6))
    plt.plot(test_predictions, label='Predicted')
    plt.plot(test_data[1:], label='Actual')
    plt.xlabel('Sequences')
    plt.ylabel(target)
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'sarima_lstm_test_pred.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 计算误差
    testScore1 = math.sqrt(mean_squared_error(test_data[1:], test_predictions))
    print('Test Score: %.2f RMSE' % (testScore1))
    testScore2 = mean_absolute_error(test_data[1:], test_predictions)
    print('Test Score: %.2f MAE' % (testScore2))
    testScore3 = r2_score(test_data[1:], test_predictions)
    print('Test Score: %.2f R2' % (testScore3))
    testScore4 = mean_absolute_percentage_error(test_data[1:], test_predictions)
    print('Test Score: %.2f MAPE' % (testScore4))

    trainScore1 = math.sqrt(mean_squared_error(train_data[1:], train_predictions))
    print('train Score: %.2f RMSE' % (trainScore1))
    trainScore2 = mean_absolute_error(train_data[1:], train_predictions)
    print('train Score: %.2f MAE' % (trainScore2))
    trainScore3 = r2_score(train_data[1:], train_predictions)
    print('train Score: %.2f R2' % (trainScore3))
    trainScore4 = mean_absolute_percentage_error(train_data[1:], train_predictions)
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

    df.to_excel(os.path.join(base_dir, 'result', 'sarima_lstm_eval.xlsx'), index=False)   # 将评估指标值存为数据表


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    target = '实际质量值'
    # filename = 'my_data.csv'
    filename = 'my_data_enhance_1h.csv'
    df = read_data(filename)
    lstm_model(df)
    sarima_lstm(df)

