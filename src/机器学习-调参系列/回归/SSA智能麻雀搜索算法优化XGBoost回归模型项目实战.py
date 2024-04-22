#!/usr/bin/env python
# coding: utf-8

"""
作者：胖哥
微信公众号：胖哥真不错
微信号：zy10178083

为了防止大家在运行项目时报错(项目都是运行好的，报错基本都是版本不一致 导致的)，
胖哥把项目中用到的库文件版本在这里说明：

pandas == 1.1.5
matplotlib == 3.3.4
seaborn == 0.11.1
scikit-learn == 0.24.1
numpy == 1.19.5

"""

# 导入第三方库
from sklearn.model_selection import train_test_split  # 数据集拆分工具
import pandas as pd  # 数据处理库
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 数据可视化库
import seaborn as sns  # 导入数据集分布可视化库  seaborn是基于matplotlib的数据集分布可视化库。
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  # 模型评估方法
import warnings  # 告警库
from xgboost import XGBRegressor  # 导入XGBoost回归器
from sklearn.model_selection import cross_val_score  # 交叉验证

warnings.filterwarnings(action='ignore')  # 忽略告警


# 定义适应度计算函数
def fitness_spaction(parameter, X_train, X_test, y_train, y_test):
    """
    :param parameter: 参数
    :return: 适应度
    """

    if int(abs(parameter[0])) > 0:  # 判断取值
        n_estimators = int(abs(parameter[0])) + 100  # 赋值
    else:
        n_estimators = int(abs(parameter[0])) + 100  # 赋值

    if int(abs(parameter[1])) > 0:  # 判断取值
        learning_rate = (int(abs(parameter[1]))/1000 + 1) / 10  # 赋值
    else:
        learning_rate = (int(abs(parameter[1]))/1000 + 1) / 10  # 赋值

    # 建立XGBoost模型并训练
    xgbr_model = XGBRegressor(n_estimators=n_estimators,
                              learning_rate=learning_rate, use_label_encoder=False,
                              eval_metric=['logloss']).fit(
        X_train, y_train)  # 建模、拟合
    cv_accuracies = cross_val_score(xgbr_model, X_test, y_test, cv=3,
                                    scoring='r2')  # 交叉验证计算r方

    # 使错误率降到最低
    accuracies = cv_accuracies.mean()  # 取交叉验证均值

    # 使错误率降到最低
    fitness_value = (1 - accuracies)  # 错误率 赋值 适应度函数值

    return fitness_value  # 返回适应度


# 定义边界函数
def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:  # 小于最小值
            temp[i] = Lb[0, i]  # 取最小值
        elif temp[i] > Ub[0, i]:  # 大于最大值
            temp[i] = Ub[0, i]  # 取最大值

    return temp  # 返回数据


# 定义麻雀智能优化算法
def SSA(pop, M, c, d, dim, fun, X_train, X_test, y_train, y_test):
    """
    :param fun: 适应度函数
    :param pop: 种群数量
    :param M: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :param dim: 优化参数的个数
    :return: 适应度值最小的值 对应得位置
    """
    P_percent = 0.2  # 种群初始比例
    pNum = round(pop * P_percent)  # 种群初始化
    lb = c * np.ones((1, dim))  # 最小限制
    ub = d * np.ones((1, dim))  # 最大限制
    X = np.zeros((pop, dim))  # 初始位置
    fit = np.zeros((pop, 1))  # 初始适应度

    # 种群循环
    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)  # 位置
        fit[i, 0] = fun(X[i, :], X_train, X_test, y_train, y_test)  # 调用适应度函数计算适应度
    pFit = fit  # 赋值
    pX = X  # 赋值
    fMin = np.min(fit[:, 0])  # 适应度最小值
    bestI = np.argmin(fit[:, 0])  # 适应度最小值
    bestX = X[bestI, :]  # 最优位置
    Convergence_curve = np.zeros((1, M))  # 收敛曲线
    # 进行迭代
    for t in range(M):
        print('*************************正在迭代第', t + 1, '次**********************************')
        sortIndex = np.argsort(pFit.T)  # 适应度排序
        fmax = np.max(pFit[:, 0])  # 适应度最大值
        B = np.argmax(pFit[:, 0])  # 适应度最大值
        worse = X[B, :]  # 最差的适应度值
        # 发现者位置更新
        r2 = np.random.rand(1)  # 生成0~1之间的随机数
        if r2 < 0.8:  # 预警值和安全值为0.8  此时的觅食环境周围没有捕食者，发现者可以执行广泛的搜索操作
            # 循环
            for i in range(pNum):
                r1 = np.random.rand(1)  # 生成0~1之间的随机数
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i) / (r1 * M))  # 位置
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)  # 位置边界处理
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :], X_train, X_test, y_train, y_test)  # 适应度
        elif r2 >= 0.8:  # 预警值和安全值为0.8 这表示种群中的一些麻雀已经发现了捕食者，并向种群中其它麻雀发出了警报，此时所有麻雀都需要迅速飞到其它安全的地方进行觅食。
            # 循环
            for i in range(pNum):
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] + np.random.rand(1) * np.ones((1, dim))  # 位置
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)  # 位置边界处理
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :], X_train, X_test, y_train, y_test)  # 适应度
        bestII = np.argmin(fit[:, 0])  # 最小适应度
        bestXX = X[bestII, :]  # 最优位置
        # 加入者位置更新
        for ii in range(pop - pNum):
            i = ii + pNum
            A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1  # A赋值
            if i > pop / 2:  # 当i >pop/2时，这表明，适应度值较低的第i个加入者没有获得食物，处于十分饥饿的状态，此时需要飞往其它地方觅食，以获得更多的能量。
                X[sortIndex[0, i], :] = np.random.rand(1) * np.exp(worse - pX[sortIndex[0, i], :] / np.square(i))  # 位置
            else:
                X[sortIndex[0, i], :] = bestXX + np.dot(np.abs(pX[sortIndex[0, i], :] - bestXX),
                                                        1 / (A.T * np.dot(A, A.T))) * np.ones((1, dim))  # 位置
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)  # 位置边界处理
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :], X_train, X_test, y_train, y_test)  # 适应度
        arrc = np.arange(len(sortIndex[0, :]))  # 返回步长为1的数组

        # 意识到危险得麻雀位置更新
        c = np.random.permutation(arrc)  # 随机排列序列
        b = sortIndex[0, c[0:20]]
        # 循环
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:  # 当fi >fg表示此时的麻雀正处于种群的边缘，极其容易受到捕食者的攻击。
                X[sortIndex[0, b[j]], :] = bestX + np.random.rand(1, dim) * np.abs(
                    pX[sortIndex[0, b[j]], :] - bestX)  # 位置
            else:  # fi = fg时，这表明处于种群中间的麻雀意识到了危险，需要靠近其它的麻雀以此尽量减少它们被捕食的风险。
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 10 ** (-50))  # 位置
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)  # 位置边界处理
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]], X_train, X_test, y_train, y_test)  # 适应度
        # 循环
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]  # 适应度
                pX[i, :] = X[i, :]  # 位置
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]  # 适应度
                bestX = pX[i, :]  # 位置
        Convergence_curve[0, t] = fMin  # 适应度
    return fMin, bestX, Convergence_curve  # 最小适应度  最优位置  收敛曲线


if __name__ == "__main__":
    # 读取数据
    data = pd.read_excel('data.xlsx')

    # 用Pandas工具查看数据
    print(data.head())

    # 数据缺失值统计
    print('****************************************')
    print(data.info())

    # 数据描述性统计分析
    print('****************************************')
    print(data.describe().round(4))  # 保留4位小数点

    # y变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = data['y']  # 过滤出y变量的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('y')  # 设置x轴名称
    plt.ylabel('数量')  # 设置y轴名称
    plt.title('y变量分布直方图')  # 设置标题名称
    plt.show()  # 展示图片

    # 数据的相关性分析
    sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')  # 设置标题名称
    plt.show()  # 展示图片

    # 构建特征和标签
    X = data.drop(columns=['y'])  # 构建特征
    y = data['y']  # 构建标签

    # 数据集的划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SSA初始化参数
    SearchAgents_no = 10  # 种群数量
    Max_iteration = 1  # 迭代次数
    dim = 2  # 优化参数的个数
    lb = [10 ** (-1), 2 ** (-5)]  # 最小值限制
    ub = [10 ** 1, 2 ** 4]  # 最大值限制

    # 调用麻雀智能优化算法
    fMin, bestX, SSA_curve = SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction, X_train, X_test,
                                 y_train, y_test)

    if int(abs(bestX[0])) > 0:  # 判断
        best_n_estimators = int(abs(bestX[0])) + 500  # 赋值
    else:
        best_n_estimators = int(abs(bestX[0])) + 1000  # 赋值

    if int(abs(bestX[1])) > 0:  # 判断
        best_learning_rate = (int(abs(bestX[1]) / 1000) + 1) / 10  # 赋值
    else:
        best_learning_rate = (int(abs(bestX[1]) / 1000) + 1) / 10  # 赋值

    print('----------------SSA智能麻雀搜索算法优化XGBoost回归模型-最优结果展示-----------------')
    print("The best n_estimators is " + str(abs(best_n_estimators)))
    print("The best learning_rate is " + str(abs(best_learning_rate)))

    # 应用优化后的最优参数值构建XGBoost回归模型
    xgbr_model = XGBRegressor(n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                              use_label_encoder=False, eval_metric=['logloss'])  # 建模
    xgbr_model.fit(X_train, y_train)  # 拟合
    y_pred = xgbr_model.predict(X_test)  # 预测
    # 应用10折交叉验证 计算模型得分
    accuracies = cross_val_score(xgbr_model, X=X_train, y=y_train, cv=3)
    accuracy_mean = accuracies.mean()  # 取3次交叉验证均值

    print('----------------模型评估-----------------')
    # 模型评估

    print('XGBoost回归模型-最优参数-R方值：{}'.format(round(r2_score(y_test, y_pred), 4)))
    print('XGBoost回归模型-最优参数-均方误差：{}'.format(round(mean_squared_error(y_test, y_pred), 4)))
    print('XGBoost回归模型-最优参数-可解释方差值：{}'.format(round(explained_variance_score(y_test, y_pred), 4)))
    print('XGBoost回归模型-最优参数-平均绝对误差：{}'.format(round(mean_absolute_error(y_test, y_pred), 4)))

    # 真实值与预测值比对图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")  # 绘制折线图
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")  # 绘制折线图
    plt.legend(['真实值', '预测值'])  # 设置图例
    plt.title("SSA智能麻雀搜索算法优化XGBoost回归模型真实值与预测值比对图")  # 设置标题名称
    plt.show()  # 显示图片
