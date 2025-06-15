import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt


'''
yfinance是Ran Aroussi开发的一个流行的开源库，用于访问雅虎财经上提供的财务数据。
'''

# 获取股票数据
def load_data():
    data = pd.read_csv('data.csv')
    # 将日期列转换为日期时间类型
    data['Month'] = pd.to_datetime(data['Month'])
    # 将日期列设置为索引
    data.set_index('Month', inplace=True)
    return data

# 数据预处理
def preprocess_data(data, n_steps=50):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i + n_steps])
        y.append(scaled_data[i + n_steps])
    return np.array(X), np.array(y), scaler

# 构建并训练 LSTM 模型
def train_lstm_model(units, learning_rate, batch_size):
    units = int(units)
    batch_size = int(batch_size)
    
    # 数据加载和预处理
    stock_data = load_data()
    X, y, scaler = preprocess_data(stock_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
passengers_data = load_data()
print(passengers_data)
X, y, scaler = preprocess_data(passengers_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final_model = Sequential([
    LSTM(best_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])
final_model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='mse')
final_model.fit(X_train, y_train, epochs=20, batch_size=best_batch_size, verbose=1)

# 模型预测
predictions = final_model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
actual_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# 绘制预测图
plt.figure(figsize=(12, 6))
plt.plot(range(len(actual_rescaled)), actual_rescaled, label="Actual Passenger Count", linestyle="-")
plt.plot(range(len(predictions_rescaled)), predictions_rescaled, label="Predicted Passenger Count", linestyle="--")
plt.title("International Airline Passengers - Actual vs Predicted")
plt.xlabel("Year")
plt.ylabel("Passenger Count")
plt.legend()
plt.show()