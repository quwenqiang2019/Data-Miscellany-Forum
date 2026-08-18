# 假设我有一个函数（原函数），可以实现两个数相除
def divide(x, y):
    return x / y

res = divide(10, 2)
print(res)

# 定义一个装饰器，使得原函数能够有处理异常的功能
def handle_exceptions(func):    # 外层函数：接受一个函数（原函数）作为参数，并返回一个函数（新函数）。
    def wrapper(*args, **kwargs):  # 里层函数：接受和原函数相同的参数
        try:
            # 调用被装饰的函数
            return func(*args, **kwargs)     # 将原来的函数包装在自己功能中，同时添加新的功能
        except Exception as e:
            # 处理异常的逻辑
            # 返回异常信息
            return e
    return wrapper   # 新函数


# 使用@符号+装饰器的名字来使用装饰器，可以在不改变原函数（divide）的情况下增加异常处理的功能
@handle_exceptions
def divide(x, y):    # 这是想要装饰的函数（原函数），即实现两个数相除
    return x / y

# 也可以使用divide = handle_exceptions(divide)来使用装饰器，可以在不改变原函数（divide）的情况下增加异常处理的功能
# divide = handle_exceptions(divide)

res = divide(10, 0)
print(res)