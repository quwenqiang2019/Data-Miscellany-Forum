import numpy as np

# 二维数组
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# 法1：将二维数组转换为一维数组
array_1d = array_2d.flatten()
print('法1：', array_1d)

# 法2：将二维数组转换为一维数组
array_1d = array_2d.reshape(-1)
print('法2：', array_1d)

# 法3：将二维数组转换为一维数组
array_1d = np.array([i for item in array_2d for i in item])
print('法3：', array_1d)
