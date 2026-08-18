import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 生成虚拟数据
np.random.seed(0)
X = np.random.rand(100, 1) * 10
print(X)
y = 2 * X**2 + np.random.randn(100, 1) * 2

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# 构建神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_scaled, y_scaled, epochs=100, batch_size=32)

# 使用模型进行预测
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
X_test_scaled = scaler.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled)

# 可视化结果
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Actual')
# plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()
