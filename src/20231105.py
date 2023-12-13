# 实例一：
class A(object):
    def __init__(self):
        print("class ---- A ----")

class B(A):
    def __init__(self):
        print("class ---- B ----")
        super(B, self).__init__()

class C(A):
    def __init__(self):
        print("class ---- C ----")
        super(C, self).__init__()


# 实例一
class D(B, C):
    def __init__(self):
        print(D.__mro__)
        print("class ---- D ----")
        super(D, self).__init__()
d = D()
'''
#输出结果：
class ---- D ----
class ---- B ----
class ---- C ----
class ---- A ----
'''

# 实例二
class D(B, C):
    def __init__(self):
        print("class ---- D ----")
        super(B, self).__init__()
d = D()
'''
#输出结果：
class ---- D ----
class ---- C ----
class ---- A ----
'''

# 实例三
class D(B, C):
    def __init__(self):
        print("class ---- D ----")
        super(C, self).__init__()
d = D()
'''
# 输出结果：
class ---- D ----
class ---- A ----
'''

