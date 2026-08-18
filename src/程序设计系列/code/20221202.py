#=======多位列表（数组）转化为一维列表（数组）==================
# 方法1:利用数组的flatten
import numpy as np
mulArrays = [[1,2,3],[4,5,6],[7,8,9]]
print(list(np.array(mulArrays).flatten()))
# 方法2:列表推导式
mulArrays = [[1,2,3],[4,5,6],[7,8,9]]
print([i for arr in mulArrays for i in arr])

#=======一维列表（数组）转化为多维列表（数组）==================
# 方法1:利用numpy
import numpy as np
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Y = np.array(x).reshape(3, 3)
X=[]
for i in Y:
    X.append(list(i))
print(X)
# 方法2:叠加法自定义各个维度
s = [1, 2, 3, 4, 5, 6, 7, 8, 9]
x = 2;y = 2;z = 3
a = []; b = [];c = []
for i in s:
    if len(a) < x:
        a.append(i)
        continue
    if len(b) < y:
        b.append(a)
        a = []
        b.append(i)
    else:
        continue
    if len(c) < z:
        c.append(b)
        b = []
    else:
        continue
print(c)

