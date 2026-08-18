from sklearn.linear_model import LinearRegression
import numpy as np


# 计算相邻两个数之间的差值的均值，并判断变化趋势。
def trend(lst):
    diff = [lst[i + 1] - lst[i] for i in range(len(lst) - 1)]
    trend = sum(diff) / len(diff)
    if trend > 0:
        return '上升'
    elif trend < 0:
        return '下降'
    else:
        return '不明显'


# 使用线性回归模型来拟合数据，并根据回归系数的正负来判断整体变化趋势。
def lr_trend(lst):
    # 训练线性回归模型
    reg = LinearRegression().fit(np.arange(len(lst)).reshape(-1, 1), lst)
    # 判断整体变化趋势
    if reg.coef_ > 0:
        return '上升'
    elif reg.coef_ < 0:
        return '下降'
    else:
        return '不明显'


lst = [453, 443, 388, 454, 366]
t = trend(lst)
print(t)
t = lr_trend(lst)
print(t)