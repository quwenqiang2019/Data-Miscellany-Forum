# 导入第三方库
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# 定义数据加载、数据预处理、探索性数据分析函数
def import_data():
    """
    :return: 训练集和测试集数据
    """
    data = pd.read_csv('Dataset.csv')
    data = shuffle(data)
    # 构建特征和标签
    X = data.drop(columns=['target'])  # 构建特征
    y = data['target']  # 构建标签

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 数据集拆分

    return x_train, x_test, y_train, y_test  # 返回训练集和测试集


# 定义Levy飞行策略函数
def levy_flight(x, a):
    return np.where(x > 0, (x + a) ** 2 / (4 * a), x ** 2 / (4 * a))  # 返回数值


# 定义反向学习函数
def backward_learning(x, y, learning_rate=0.1):
    y_pred = levy_flight(x, a=0.1)  # 调用Levy飞行策略函数

    loss = np.mean((y - y_pred) ** 2)  # 损失

    grad = -2 * (y - y_pred) / len(y)  # 计算

    return loss, grad  # 返回数值


# 定义适应度计算函数
def fitness_spaction(parameter):
    """
    :param parameter: SVM参数
    :return: 最小化错误率
    """
    data_train, data_test, label_train, label_test = import_data()  # 调用数据加载函数
    # SVM参数
    c = parameter[0]
    g = parameter[1]
    # 训练测试
    clf = SVC(gamma=g, C=c)  # 建模
    clf.fit(data_train, label_train)  # 拟合
    y_predict = clf.predict(data_test)  # 预测
    acc = accuracy_score(label_test, y_predict)  # 计算准确率

    return 1 - acc  # 返回错误率


# 定义边界函数
def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:  # 小于最小值
            temp[i] = Lb[0, i]  # 取最小值
        elif temp[i] > Ub[0, i]:  # 大于最大值
            temp[i] = Ub[0, i]  # 取最大值

    return temp


# 定义融合反向学习与Levy飞行策略的改进麻雀优化算法ISSA
def ISSA(pop, M, c, d, dim, fun):
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

    x = np.random.uniform(-10, 10, size=(M, 1))  # 生成随机数
    y = np.array([levy_flight(xi, a=0.1) for xi in x])  # 生成数据y

    # 随机选择两个点进行反向学习
    idx1 = random.randint(0, len(x) - 1)  # 生成随机数
    idx2 = random.randint(0, len(x) - 1)  # 生成随机数

    if idx1 != idx2:  # 判断
        loss, grad = backward_learning(x[idx1], y[idx1], learning_rate=0.1)  # 调用定义反向学习函数

        x[idx1] += 0.1 * grad  # 赋值

        x[idx2] += 0.1 * grad  # 赋值

    loss, _ = backward_learning(x[-1], y[-1])  # 调用定义反向学习函数

    # 种群循环
    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)  # 位置
        fit[i, 0] = fun(X[i, :]) + loss  # 调用适应度函数计算适应度
    pFit = fit
    pX = X
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
        v0 = 0.5
        v = v0 + t / (3 * M)
        if r2 < 0.8:  # 预警值和安全值为0.8  此时的觅食环境周围没有捕食者，发现者可以执行广泛的搜索操作
            # 循环
            for i in range(pNum):
                r1 = np.random.rand(1)  # 生成0~1之间的随机数
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i) / (r1 * M))  # 位置
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)  # 位置边界处理
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])  # 适应度
        elif r2 >= 0.8:  # 预警值和安全值为0.8 这表示种群中的一些麻雀已经发现了捕食者，并向种群中其它麻雀发出了警报，此时所有麻雀都需要迅速飞到其它安全的地方进行觅食。
            # 循环
            for i in range(3, pNum):  # 分数阶
                X[sortIndex[0, 0], :] = pX[sortIndex[0, 0], :] * v + np.random.rand(1) * np.ones((1, dim))  # 位置
                X[sortIndex[0, 1], :] = pX[sortIndex[0, 1], :] * v - 1 / 2 * v * (v - 1) * X[
                    sortIndex[0, 0]] + np.random.rand(1) * np.ones((1, dim))  # 位置
                X[sortIndex[0, 2], :] = pX[sortIndex[0, 2], :] * v - 1 / 2 * v * (v - 1) * X[
                    sortIndex[0, 1]] - 1 / 6 * v * (v - 1) * (v - 2) * X[sortIndex[0, 0]] + np.random.rand(1) * np.ones(
                    (1, dim))  # 位置
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * v - 1 / 2 * v * (v - 1) * X[
                    sortIndex[0, i - 1]] - 1 / 6 * v * (v - 1) * (v - 2) * X[sortIndex[0, i - 2]] + 1 / 24 * v * (
                                                v - 1) * (v - 2) * (v - 3) * X[
                                            sortIndex[0, i - 3]] + np.random.rand(1) * np.ones((1, dim))  # 位置

                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)  # 位置边界处理
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])  # 适应度
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
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])  # 适应度
        arrc = np.arange(len(sortIndex[0, :]))  # 返回步长为1的数组

        # 意识到危险得麻雀位置更新
        c = np.random.permutation(arrc)  # 随机排列序列
        b = sortIndex[0, c[0:20]]
        # 循环
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:  # 当fi >fg表示此时的麻雀正处于种群的边缘，极其容易受到捕食者的攻击。
                X[sortIndex[0, b[j]], :] = bestX + np.random.rand(1, dim) * np.abs(
                    pX[sortIndex[0, b[j]], :] - bestX)  # 位置
            else:
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 10 ** (-50))  # 位置
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)  # 位置边界处理
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]])  # 适应度

        # 混合变异策略
        y1 = 1 - t ** 2 / M ** 2
        y2 = t ** 2 / M ** 2
        sj = random.random()
        coefficient = y1 / (math.pi * (1 + sj ** 2)) + y2 * math.exp(-sj ** 2 / 2) / math.sqrt(2 * math.pi)

        bestXF = bestX + coefficient  # 位置
        if fun(bestXF) < fMin:  # 适应度判断
            fMin = fun(bestXF)  # 适应度小于最小值  进行赋值
            bestX = bestXF  # 位置

        # 种群循环
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:  # 适应度判断
                pFit[i, 0] = fit[i, 0]  # 适应度
                pX[i, :] = X[i, :]  # 位置
            if pFit[i, 0] < fMin:  # 适应度判断
                fMin = pFit[i, 0]  # 适应度小于最小值  进行赋值
                bestX = pX[i, :]  # 位置

        Convergence_curve[0, t] = fMin  # 收敛曲线
    return fMin, bestX, Convergence_curve  # 最小适应度  最优位置  收敛曲线


if __name__ == "__main__":
    data_train, data_test, label_train, label_test = import_data()

    # 未使用改进的麻雀优化算法进行训练预测
    clf = SVC(probability=True)  # 建模
    clf.fit(data_train, label_train)  # 拟合
    y_predict = clf.predict(data_test)  # 预测

    acc = accuracy_score(label_test, y_predict)  # 计算准确率
    p = precision_score(label_test, y_predict)  # 计算查准率
    r = recall_score(label_test, y_predict)  # 计算查全率
    f = f1_score(label_test, y_predict)  # 计算F1分值
    print('*********************未使用改进的麻雀优化算法模型评估**************************')
    print('SVM 准确率: {0}'.format(acc))
    print("SVM 查准率: {0}".format(p))
    print("SVM 查全率: {0}".format(r))
    print("SVM F1分值: {0}".format(f))
    print('******************************************************************')

    # SSA初始化参数
    SearchAgents_no = 30  # 种群数量
    Max_iteration = 10  # 迭代次数
    dim = 2  # 优化参数的个数
    lb = [10 ** (-1), 2 ** (-5)]  # 最小值限制
    ub = [10 ** 1, 2 ** 4]  # 最大值限制
    # lb = [0.1, 10]  # 最小值限制
    # ub = [1, 50]  # 最大值限制
    print(lb)
    print(ub)

    # 调用改进的麻雀智能优化算法
    fMin, bestX, SSA_curve = ISSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    print('******************************ISSA 准确率********************************')
    print('ISSA 准确率:{0}'.format(1 - fMin))
    print('******************************c和g的最优值********************************')
    print("c: {0}, g: {1}".format(bestX[0], bestX[1]))
    print('\n')
    # 应用最优值进行建模
    clf = SVC(gamma=bestX[1], C=bestX[0])  # 建模
    clf.fit(data_train, label_train)  # 拟合
    y_predict = clf.predict(data_test)  # 预测
    acc = accuracy_score(label_test, y_predict)  # 准确率
    p = precision_score(label_test, y_predict)  # 计算查准率
    r = recall_score(label_test, y_predict)  # 计算查全率
    f = f1_score(label_test, y_predict)  # 计算F1分值
    print('******************************使用ISSA改进的麻雀优化算法的模型评估************************************')
    print("ISSA-SVM 准确率: {0}".format(acc))
    print("ISSA-SVM 查准率: {0}".format(p))
    print("ISSA-SVM 查全率: {0}".format(r))
    print("ISSA-SVM F1分值: {0}".format(f))
    print('******************************************************************************************')

    # 查看是否过拟合
    print('训练集score: {:.4f}'.format(clf.score(data_train, label_train)))
    print('测试集score: {:.4f}'.format(clf.score(data_test, label_test)))

    from sklearn.metrics import classification_report  # 导入分类报告工具

    # 分类报告
    print(classification_report(label_test, y_predict))

    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵工具
    import seaborn as sns  # 统计数据可视化

    # 混淆矩阵
    cm = confusion_matrix(label_test, y_predict)
    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])

    # 构建数据框
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual :0', 'Actual :1'],
                             index=['Predict :0', 'Predict :1'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')  # 热力图展示
    plt.show()  # 展示图片
