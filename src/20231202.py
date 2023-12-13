import numpy as np
from sklearn.linear_model import LinearRegression

# 假设我们有三个自变量 X1、X2 和 X3，以及一个因变量 y
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([10, 20, 30])

# 创建一个线性回归模型对象
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 输出模型的系数和截距
print(model.coef_)
print(model.intercept_)

# 预测新的数据点
new_X = np.array([[2, 3, 4]])
print(model.predict(new_X))



# 假设我们有一个自变量 X1，以及一个因变量 y
X = np.array([[1], [4], [7]])
y = np.array([[10], [20], [30]])

# 创建一个线性回归模型对象
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 输出模型的系数和截距
print(model.coef_)
print(model.intercept_)

# 预测新的数据点
new_X = np.array([[3]])
print(model.predict(new_X))
