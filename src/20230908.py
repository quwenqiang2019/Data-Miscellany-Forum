# 下面是一个使用map()函数的例子，将一个列表中的所有元素都乘以2：
numbers = [1, 2, 3, 4, 5]

def double(x):
    return x * 2

result = map(double, numbers)

print(list(result))  # 输出 [2, 4, 6, 8, 10]


# map()函数还可以应用于多个可迭代对象，例如：
numbers1 = [1, 2, 3, 4, 5]
numbers2 = [10, 20, 30, 40, 50]

def add(x, y):
    return x + y

result = map(add, numbers1, numbers2)

print(list(result))  # 输出 [11, 22, 33, 44, 55]