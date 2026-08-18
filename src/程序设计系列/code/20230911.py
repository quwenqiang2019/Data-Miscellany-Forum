from functools import reduce

numbers = [1, 2, 3, 4, 5]

def multiply(x, y):
    return x * y

result = reduce(multiply, numbers)
print(result)  # 输出 120


numbers = [1, 2, 3, 4, 5]

def multiply(x, y):
    return x * y

result = reduce(multiply, numbers, 2)
print(result)  # 输出 240

