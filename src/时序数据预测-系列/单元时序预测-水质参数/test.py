import numpy as np

# 创建一个形状为（10，1）的数组
array1 = np.random.randint(0, 10, size=(10, 1))

# 创建一个形状为（10，4）的数组
array2 = np.random.randint(0, 10, size=(10, 4))

# 将两个数组进行拼接，沿第二个维度（axis=1）拼接
result = np.concatenate((array1, array2), axis=1)

print("形状为（10，1）的数组：")
print(array1)

print("\n形状为（10，4）的数组：")
print(array2)

print("\n拼接后形状为（10，5）的数组：")
print(result)
