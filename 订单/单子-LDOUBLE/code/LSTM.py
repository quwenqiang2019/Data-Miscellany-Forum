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


def read_data(filename):
    # 读取数据集
    df = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', filename)))
    df = df.rename(columns={'日期':'年份', 'Unnamed: 1':'月份', 'Unnamed: 2':'日期'})
    # 将年份、月份和日期列组合成表示日期格式的一列
    df['date'] = pd.to_datetime(df['年份'].astype(str) + '-' + df['月份'].astype(str) + '-' + df['日期'].astype(str))
    df = df.drop(columns=['年份', '月份', '日期', '线性插值'])
    # 将日期列设置为索引
    df.set_index('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)

    return df




def nan_insert(df):
    df['人口'].fillna(df['人口'].mean(), inplace=True)
    df1 = df.iloc[:, 1:]
    train_df1 = df1[df1['未插值'].notna()]
    test_df1 = df1[df1['未插值'].isna()]
    # 特征和标签
    X_train = train_df1.drop(columns=['未插值'])
    y_train = train_df1['未插值']
    X_test = test_df1.drop(columns=['未插值'])
    # 训练随机森林模型
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    # 预测缺失值
    y_pred = rf_model.predict(X_test)
    # 填补缺失值
    df.loc[df['未插值'].isna(), '未插值'] = y_pred
    print("\n填补后的数据:")
    print(df)

    df.to_excel(os.path.join(base_dir, 'result', 'insert_data.xlsx'), index=False)  # 将填补后的数据存为数据表

    return df



def lstm_model(df):
    fea_num = len(df.columns)
    # 数据划分
    test_split=round(len(df)*0.20)
    df_for_training=df[:-test_split]
    df_for_testing=df[-test_split:]
    print(df)
    print(df_for_training)
    print(df_for_testing)
    # 绘制训练集和测试集的折线图
    # 可视化部分
    sns.set(font_scale=1.2)
    plt.rc('font', family=['SimSun'], size=12)
    # plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(10, 6))
    plt.plot(df_for_training['真值'], label='训练集')
    plt.plot(df_for_testing['真值'], label='数据集')
    plt.xlabel('时间序列')
    plt.ylabel('真值')
    plt.title('数据集')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'data_split.jpg'), bbox_inches='tight', dpi=600)
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

    window_size = 1
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
    my_model.add(LSTM(50, return_sequences=True))
    my_model.add(Dropout(0.2))
    my_model.add(LSTM(50))
    my_model.add(Dropout(0.2))
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
    df_train = pd.DataFrame({'pred': pred_train, 'original': original_train})
    df_train.to_excel(os.path.join(base_dir, 'result', 'train_pred.xlsx'), index=False)  # 将评估指标值存为数据表

    plt.plot(df_for_training.index[window_size:,], original_train, color = 'red', label = '真实值')
    plt.plot(df_for_training.index[window_size:,], pred_train, color = 'blue', label = '预测值')
    plt.title('真值预测')
    plt.xlabel('时间序列')
    plt.xticks(rotation=45)
    plt.ylabel('真值')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'train_pred.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


    prediction_test_copies_array = np.repeat(prediction_test,fea_num, axis=-1)
    pred_test=scaler.inverse_transform(np.reshape(prediction_test_copies_array,(len(prediction_test),fea_num)))[:,0]
    original_test_copies_array = np.repeat(testY, fea_num, axis=-1)
    original_test=scaler.inverse_transform(np.reshape(original_test_copies_array,(len(testY),fea_num)))[:,0]
    print("test Pred Values-- ", pred_test)
    print("\ntest Original Values-- ", original_test)
    df_test = pd.DataFrame({'pred': pred_test, 'original': original_test})
    df_test.to_excel(os.path.join(base_dir, 'result', 'test_pred.xlsx'), index=False)  # 将评估指标值存为数据表

    plt.plot(df_for_testing.index[window_size:,], original_test, color = 'red', label = '真实值')
    plt.plot(df_for_testing.index[window_size:,], pred_test, color = 'blue', label = '预测值')
    plt.title('真值预测')
    plt.xlabel('时间序列')
    plt.xticks(rotation=45)
    plt.ylabel('真值')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'result', 'test_pred.jpg'), bbox_inches='tight', dpi=600)
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

    df.to_excel(os.path.join(base_dir, 'result', 'eval.xlsx'), index=False)   # 将评估指标值存为数据表


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    filename = '数据.xlsx'
    df = read_data(filename)
    df = nan_insert(df)
    lstm_model(df)

