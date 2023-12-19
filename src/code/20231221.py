import numpy as np

X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
X2 = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])


print('元素数量', X2.size)  # 输出数组元素的个数
print('行数', np.size(X2, 0), '列数', np.size(X2, 1)) # 输出行数和列数
print("维度:", X2.shape)   # 输出数组的形状（维度）
print('行数', X2.shape[0], '列数', X2.shape[1])  # 输出行数和列数
print('长度', len(X2))  # 输出数组的长度

X3 = X2.reshape(len(X1), 1)
X4 = np.reshape(X1, (-1, 1))
print(X3, X4)

