# 使用all()函数判断列表中所有元素是否为偶数
numbers = [2, 4, 6, 8, 10]
result = all(num % 2 == 0 for num in numbers)
print(result)
# 输出: True

# 使用any()函数判断列表中是否存在奇数
numbers = [2, 4, 6, 8, 9, 10]
result = any(num % 2 == 1 for num in numbers)
print(result)
# 输出: True
