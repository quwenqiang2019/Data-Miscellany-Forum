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

X3 = X1.reshape(len(X1), 1)
X4 = np.reshape(X1, (-1, 1))
print(X3, X4)


X5 = X2.flatten()
print(X5)


a1 = np.array([1, 3, 4])    # shape (3,)
a2 = np.array([4, 6, 7])    # shape (3,)
a3 = np.array([8, 10, 14])    # shape (3,)
b1 = np.array([[1,2,3],[4,5,6]])  # shape (3, 3)
b2 = np.array([[11,21,31],[7,8,9]])  # shape (3, 3)


c1 = np.stack((a1, a2, a3), axis=0)
print(c1)
c2 = np.stack((a1, a2, a3), axis=1)
print(c2)

d1 = np.vstack((a1, a2, a3))
print(d1)

e1 = np.hstack((a1, a2, a3))
print(e1)

f1 = np.dstack((a1, a2, a3))
print(f1)

g1 = np.row_stack((a1, a2, a3))
print(g1)

h1 = np.column_stack((a1, a2, a3))
print(h1)

i1 = np.concatenate((a1, a2, a3), axis=0)
print(i1)
i2 = np.concatenate((b1, b2), axis=1)
print(i2)