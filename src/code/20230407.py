def square(x) :         # 计算平方数
    return x ** 2

a=map(square, [1,2,3,4,5])    # 计算列表各个元素的平方
print(a)    # 返回迭代器<map object at 0x000002027BE7FFA0>

b=list(a)   # 使用 list() 转换为列表
print(b)    #[1, 4, 9, 16, 25]

c=list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))   # 使用 lambda 匿名函数
print(c)   #[1, 4, 9, 16, 25]