import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import seaborn as sns


# 获取股票数据
def load_data():
    data = pd.read_csv("data.csv", parse_dates=["Date"], index_col=[0])
    data = pd.DataFrame(data)
    fea_num = len(data.columns)
    return data, fea_num

# 数据集划分
def split_data(data):
    # 划分训练集和测试集
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# 数据归一化
def normalize_data(train_data, test_data):
    # 将数据归一化到 0~1 范围
    scaler = MinMaxScaler()
    train_data_scaler = scaler.fit_transform(train_data)
    test_data_scaler = scaler.transform(test_data)
    return train_data_scaler, test_data_scaler, scaler

# 定义滑动窗口函数
def create_sliding_windows(data, window_size):
    dataX = []
    dataY = []
    for i in range(window_size, len(data)):
        dataX.append(data[i - window_size:i, 0:data.shape[1]])
        dataY.append(data[i, 0])
    return np.array(dataX), np.array(dataY)

# 构建并训练 LSTM 模型
def train_lstm_model(units, learning_rate, batch_size):
    units = int(units)
    batch_size = int(batch_size)
    
    # 数据加载和预处理
    stock_data, fea_num = load_data()
    train_data, test_data = split_data(stock_data)
    train_data_scaler, test_data_scaler, scaler = normalize_data(train_data, test_data)
    # 定义滑动窗口大小
    window_size = 7
    X_train, y_train = create_sliding_windows(train_data_scaler, window_size)
    X_test, y_test = create_sliding_windows(test_data_scaler, window_size)
    # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    X_train = np.reshape(X_train, (X_train.shape[0], window_size, fea_num))
    X_test = np.reshape(X_test, (X_test.shape[0], window_size, fea_num))


    # 构建 LSTM 模型
    model = Sequential([
        LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    # 模型训练
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=0)
    
    # 验证损失
    val_loss = model.evaluate(X_test, y_test, verbose=0)
    return -val_loss  # 目标是最大化，所以返回负值


if __name__ == '__main__':
    # 贝叶斯优化
    pbounds = {
        'units': (10, 100),  # LSTM隐藏单元数
        'learning_rate': (1e-4, 1e-2),  # 学习率
        'batch_size': (16, 128)  # 批量大小
    }

    optimizer = BayesianOptimization(
        f=train_lstm_model,
        pbounds=pbounds,
        random_state=42
    )

    # 执行贝叶斯优化
    optimizer.maximize(init_points=5, n_iter=10)

    # 最佳超参数
    print("最佳参数：", optimizer.max)

    # 使用最佳参数构建最终模型并预测
    best_params = optimizer.max['params']
    best_units = int(best_params['units'])
    best_learning_rate = best_params['learning_rate']
    best_batch_size = int(best_params['batch_size'])

    # 重新加载数据并训练最终模型
    stock_data, fea_num = load_data()
    train_data, test_data = split_data(stock_data)
    train_data_scaler, test_data_scaler, scaler = normalize_data(train_data, test_data)
    # 定义滑动窗口大小
    window_size = 7
    X_train, y_train = create_sliding_windows(train_data_scaler, window_size)
    X_test, y_test = create_sliding_windows(test_data_scaler, window_size)
    # 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
    X_train = np.reshape(X_train, (X_train.shape[0], window_size, fea_num))
    X_test = np.reshape(X_test, (X_test.shape[0], window_size, fea_num))


    final_model = Sequential([
        LSTM(best_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(1)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='mse')
    final_model.fit(X_train, y_train, epochs=20, batch_size=best_batch_size, verbose=1)

    # 模型预测
    prediction_train = final_model.predict(X_train)
    prediction_test = final_model.predict(X_test)

    prediction_train_copies_array = np.repeat(prediction_train, fea_num, axis=-1)
    pred_train = scaler.inverse_transform(np.reshape(prediction_train_copies_array, (len(prediction_train), fea_num)))[
                 :, 0]
    original_train_copies_array = np.repeat(y_train, fea_num, axis=-1)
    original_train = scaler.inverse_transform(np.reshape(original_train_copies_array, (len(y_train), fea_num)))[:, 0]
    print("train Pred Values-- ", pred_train)
    print("\ntrain Original Values-- ", original_train)
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.plot(train_data.index[window_size:, ], original_train, color='red', label='真实值')
    plt.plot(train_data.index[window_size:, ], pred_train, color='blue', label='预测值')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    prediction_test_copies_array = np.repeat(prediction_test, fea_num, axis=-1)
    pred_test = scaler.inverse_transform(np.reshape(prediction_test_copies_array, (len(prediction_test), fea_num)))[:,
                0]
    original_test_copies_array = np.repeat(y_test, fea_num, axis=-1)
    original_test = scaler.inverse_transform(np.reshape(original_test_copies_array, (len(y_test), fea_num)))[:, 0]
    print("test Pred Values-- ", pred_test)
    print("\ntest Original Values-- ", original_test)
    plt.plot(test_data.index[window_size:, ], original_test, color='red', label='真实值')
    plt.plot(test_data.index[window_size:, ], pred_test, color='blue', label='预测值')
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
