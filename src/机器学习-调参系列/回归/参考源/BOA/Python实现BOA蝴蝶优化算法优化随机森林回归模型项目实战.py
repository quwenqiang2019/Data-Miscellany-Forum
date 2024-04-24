# 导入第三方库
import pygmo as pg  # 多目标优化库
import matplotlib.pyplot as plt  # 数据可视化库
import seaborn as sns  # 高级数据可视化库
import warnings, pandas as pd, numpy as np  # 告警库、数据处理库、科学计算库
from sklearn.model_selection import train_test_split  # 数据集拆分工具
import random  # 随机数库
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  # 模型评估方法
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型
from sklearn.model_selection import cross_val_score  # 交叉验证

warnings.filterwarnings("ignore")  # 忽略告警


# 定义最小化问题类
class MinimizationProblem:
    """
    对要用pygmo优化器解决的探测问题进行建模
    """

    # 定义初始化方法
    def __init__(self, solution_size, lower_bound=None, upper_bound=None):
        self.dim = solution_size  # 初始化维度
        if lower_bound is not None:
            self.min_x = lower_bound  # 最小边界
        if upper_bound is not None:
            self.max_x = upper_bound  # 最大边界

        # 使用pygmo库 并行多目标优化初始化
        self.problem = pg.problem(self)

    # 定义适应度方法
    def fitness(self, X):
        fitness = sum(X)  # 计算适用度
        return [fitness]  # 返回适应度

    # 定义边界处理方法
    def get_bounds(self):
        return ([self.min_x] * self.dim,
                [self.max_x] * self.dim)  # 返回最小边界、最大边界


# 定义蝴蝶优化算法类
class BOA:
    # 定义初始化方法
    def __init__(self, gen=5, c=0.01, a=0.1, p=0.8, mu=2, max_gen=10, variant="BOA"):
        """
        参数释义:
        - gen: 进化种群的迭代数
        - c: 形态
        - a: 力量指数
        - p: 切换概率
        - max_gen: 最大代数(用于更新感觉模态)
        - variant: 指定算法类别为BOA
        """
        self.iterations = gen if gen > 1 else 1  # 初始化迭代次数
        self.c = c  # 初始化感觉形态
        self.a = a  # 初始化力量指数
        self.p = p if p >= 0 and p <= 1 else 0.8  # 初始化切换概率
        self.mu = mu
        self.variant = variant.lower()  # 初始化算法类型
        self.max_iterations = max_gen  # 初始化最大迭代次数

    # 定义算法实现方法
    def evolve(self, population, X_train, X_test, y_train, y_test):
        bounds = population.problem.get_bounds()  # 边界数值
        solutions_x = population.get_x()  # 位置
        solutions_fitness = population.get_f()  # 适应度
        solutions_IDs = population.get_ID()  # 获取位置ID
        best_id = solutions_IDs[population.best_idx()]  # 获取最优ID
        pop = dict()  # 定义种群字典
        # 循环赋值
        for id, fit, x in zip(solutions_IDs, solutions_fitness, solutions_x):
            pop[id] = {'fit': fit, 'x': x}  # 赋值
        # 开始迭代
        for i in range(self.iterations):  # 循环
            print('******************************', '当前迭代次数为：', i + 1, '******************************')
            for id in solutions_IDs:  # 循环
                x = pop[id]['x']  # 赋值

                Sol = population.champion_x  # 决策向量

                if int(abs(Sol[0])) > 0:  # 判断取值
                    max_depth = int(abs(Sol[0]) / 100) + 2  # 赋值
                else:
                    max_depth = int(abs(Sol[0]) / 100) + 5  # 赋值

                if int(abs(Sol[1])) > 0:  # 判断取值
                    n_estimators = int(abs(Sol[1])) + 100  # 赋值
                else:
                    n_estimators = int(abs(Sol[1])) + 200  # 赋值

                # 建立随机森林模型并训练
                rfc_model = RandomForestRegressor(max_depth=max_depth,
                                                  n_estimators=n_estimators).fit(
                    X_train, y_train)  # 建模、拟合
                cv_accuracies = cross_val_score(rfc_model, X_test, y_test, cv=3,
                                                scoring='r2')  # 交叉验证计算准确率

                # 使错误率降到最低
                accuracies = cv_accuracies.mean()  # 取交叉验证均值

                # 使错误率降到最低
                fitness_value = (1 - accuracies)  # 错误率 赋值 适应度函数值

                fitness = fitness_value  # 适应度赋值
                # 判断
                if fitness > 0:
                    f = self.c * (fitness ** self.a)  # 适应度赋值
                else:
                    f = self.c * ((-fitness) ** self.a)  # 适应度赋值

                # 蝴蝶移动
                r1 = random.random()  # 生成0~1之间的随机数
                r2 = random.random()  # 生成0~1之间的随机数
                # 随机数据和概率判断
                if random.random() > self.p:
                    x += f * (r1 * r2 * pop[best_id]['x'] - x)  # 位置移动向最好的蝴蝶
                else:
                    # 在附近随机找到蝴蝶
                    j = random.choice(list(pop))  # 随机选择数据
                    k = random.choice(list(pop))  # 随机选择数据
                    x += f * (r1 * r2 * pop[j]['x'] - pop[k]['x'])  # 位置赋值

                # 边界处理并评估适应度
                x = self.force_bounds(x, bounds)  # 边界处理
                new_fitness = population.problem.fitness(x)  # 适应度

                # 评估新蝴蝶并在需要时更新种群
                if new_fitness < fitness:  # 适应度判断
                    pop[id] = {'fit': new_fitness, 'x': x}
                if new_fitness < pop[best_id]['fit']:  # 适应度判断
                    best_id = id

        # 更新种群
        for i in range(len(solutions_IDs)):
            id = solutions_IDs[i]  # 位置ID
            population.set_xf(i, pop[id]['x'], pop[id]['fit'])  # 设置种群位置和适应度

        return population  # 返回算法优化结果信息

    # 定义边界处理方法
    def force_bounds(self, x, bounds):
        for k in range(len(x)):
            if x[k] < bounds[0][k]:  # 小于最小边界
                x[k] = bounds[0][k]  # 赋值
            elif x[k] > bounds[1][k]:  # 大于最大边界
                x[k] = bounds[1][k]  # 赋值
        return x


if __name__ == '__main__':
    # 读取数据
    df = pd.read_excel('data.xlsx')

    # 查看数据前5行
    print('*************查看数据前5行*****************')
    print(df.head())

    # 数据缺失值统计
    print('**************数据缺失值统计****************')
    print(df.info())

    # 描述性统计分析
    print(df.describe())
    print('******************************')

    # y变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df['y']  # 过滤出y变量的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')  # 绘图
    plt.xlabel('y')  # 设置x轴名称
    plt.ylabel('数量')  # 设置y轴名称
    plt.title('y变量分布直方图')  # 设置标题名称
    plt.show()  # 展示图片

    # 数据的相关性分析
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')  # 设置标题名称
    plt.show()  # 展示图片

    # 提取特征变量和标签变量
    y = df.y
    X = df.drop('y', axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 调用蝴蝶优化算法BOA
    # solution_size=2 指定维度或大小。
    problem = MinimizationProblem(solution_size=2, lower_bound=0, upper_bound=100)  # 调用问题类并构建对象
    seed = 25  # 种子 使得每次运行代码时，种群的初始化状态都是一致的
    # size=5：表示种群的大小
    population = pg.population(problem, size=5, seed=seed)  # 构建种群对象
    boa = BOA()  # 建立蝴蝶优化算法对象
    best_solution = boa.evolve(population, X_train, X_test, y_train, y_test)  # 调用算法方法
    best_solution = best_solution.champion_x  # 决策向量
    best_solution_inx = np.argwhere(best_solution)  # 去除0数值
    best_solution = best_solution[best_solution_inx]  # 获取去除0后的数值
    best_solution = best_solution.flatten()  # 数组展平
    best_solution = best_solution / 100  # 比例化

    if int(abs(best_solution[0])) > 0:  # 判断
        best_max_depth = int(abs(best_solution[0]) / 100) + 12  # 赋值
    else:
        best_max_depth = int(abs(best_solution[0]) / 100) + 13  # 赋值

    if int(abs(best_solution[1])) > 0:  # 判断
        best_n_estimators = int(abs(best_solution[1])) * 10 + 500  # 赋值
    else:
        best_n_estimators = int(abs(best_solution[1])) * 10 + 600  # 赋值

    print('----------------BOA蝴蝶优化算法优化随机森林回归模型-最优结果展示-----------------')
    print("The best max_depth is " + str(abs(best_max_depth)))
    print("The best n_estimators is " + str(abs(best_n_estimators)))

    # 应用优化后的最优参数值构建随机森林回归模型
    rfc_model = RandomForestRegressor(max_depth=best_max_depth, n_estimators=best_n_estimators)  # 建模
    rfc_model.fit(X_train, y_train)  # 拟合
    y_pred = rfc_model.predict(X_test)  # 预测

    print('----------------模型评估-----------------')
    # 模型评估
    print('**************************输出测试集的模型评估指标结果*******************************')

    print('随机森林回归模型-最优参数-R方值：{}'.format(round(r2_score(y_test, y_pred), 4)))
    print('随机森林回归模型-最优参数-均方误差：{}'.format(round(mean_squared_error(y_test, y_pred), 4)))
    print('随机森林回归模型-最优参数-可解释方差值：{}'.format(round(explained_variance_score(y_test, y_pred), 4)))
    print('随机森林回归模型-最优参数-平均绝对误差：{}'.format(round(mean_absolute_error(y_test, y_pred), 4)))

    # 真实值与预测值比对图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")  # 绘制折线图
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")  # 绘制折线图
    plt.legend(['真实值', '预测值'])  # 设置图例
    plt.title("BOA蝴蝶优化算法优化随机森林回归模型真实值与预测值比对图")  # 设置标题名称
    plt.show()  # 显示图片
