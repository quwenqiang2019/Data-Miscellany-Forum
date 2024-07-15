import os
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


class ACO:
    def __init__(self, parameters):
        """
        Ant Colony Optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = len(parameters[2])  # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有蚂蚁的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局蚂蚁最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            fit = self.fitness(self.pop_x[i])
            if fit > temp:
                self.g_best = self.pop_x[i]
                temp = fit

    def fitness(self, ind_var):
        """
        个体适应值计算
        """
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]
        x4 = ind_var[3]
        y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
        return y

    def update_operator(self, gen, t, t_max):
        """
        更新算子：根据概率更新下一时刻的位置
        """
        rou = 0.8  # 信息素挥发系数
        Q = 1  # 信息释放总量
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.var_num):
                pi[i] = (t_max - t[i]) / t_max
                # 更新位置
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
                            self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保护
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新t值
            t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])
            # 更新全局最优值
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t

    def main(self):
        popobj = []
        best = np.zeros((1, self.var_num))[0]
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop_x))),
                                               np.max(np.array(list(map(self.fitness, self.pop_x)))))
            else:
                tmax, t = self.update_operator(gen, t, tmax)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen)))
            print(self.g_best)
            print(self.fitness(self.g_best))
            if self.fitness(self.g_best) > self.fitness(best):
                best = self.g_best.copy()
            print('最好的位置：{}'.format(best))
            print('最大的函数值：{}'.format(self.fitness(best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(1, self.NGEN + 1)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()

if __name__ == "__main__":
    # 导入数据
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    dataset = pd.read_excel(os.path.join(base_dir, 'data', '600028.对数收益率.xlsx'))
    df = pd.DataFrame(dataset)
    df.drop('trade_date', axis=1, inplace=True)
    print(df)
    columns = df.columns

    #  划分数据集
    features = columns[1:]
    target = ['log_earn_rate']
    train_size = int(len(df) * 0.8)
    X_train = df[features][:train_size]
    X_test = df[features][train_size:]
    y_train = df[target][:train_size]
    y_test = df[target][train_size:]

    # 基础模型 建模
    model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # # aco优化RF参数 建模
    # UP = [10, 10, 5, 5]
    # DOWN = [1, 1, 1, 1]
    # NGEN = 100
    # popsize = 100
    # parameters = [NGEN, popsize, DOWN, UP]
    # aco = ACO(parameters)
    # aco.main()
    # best_max_depth = int(aco.g_best[0])
    # best_n_estimators = int(aco.g_best[1])
    # best_min_samples_split = int(aco.g_best[2])
    # best_min_samples_leaf = int(aco.g_best[3])
    # print(best_max_depth, best_n_estimators, best_min_samples_split, best_min_samples_leaf)
    #
    # model = RandomForestRegressor(max_depth=best_max_depth, n_estimators=best_n_estimators, min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf)  # 建模
    # model.fit(X_train, y_train)  # 拟合
    # y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)



    # 可视化部分
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)

    # 训练集预测值与真实值的对比
    plt.plot(list(range(0,len(X_train))),y_train,marker='o')
    plt.plot(list(range(0,len(X_train))),y_train_pred,marker='*')
    plt.legend(['真实值','预测值'])
    plt.xlabel('序列')
    plt.ylabel('log_earn_rate')
    plt.title('训练集预测值与真实值的对比')
    plt.savefig(os.path.join(base_dir, 'result', 'RF_train.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 验证集预测值与真实值的对比
    plt.plot(list(range(0,len(X_test))),y_test,marker='o')
    plt.plot(list(range(0,len(X_test))),y_test_pred,marker='*')
    plt.legend(['真实值','预测值'])
    plt.xlabel('序列')
    plt.ylabel('log_earn_rate')
    plt.title('验证集预测值与真实值的对比')
    plt.savefig(os.path.join(base_dir, 'result', 'RF_test.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 评价指标
    trainScore1 = math.sqrt(mean_squared_error(y_train, y_train_pred))
    print('Train Score: %.2f RMSE' % (trainScore1))
    testScore1 = math.sqrt(mean_squared_error(y_test, y_test_pred))
    print('Test Score: %.2f RMSE' % (testScore1))

    trainScore2 = mean_absolute_error(y_train, y_train_pred)
    print('Train Score: %.2f MAE' % (trainScore2))
    testScore2 = mean_absolute_error(y_test, y_test_pred)
    print('Test Score: %.2f MAE' % (testScore2))

    trainScore3 = r2_score(y_train, y_train_pred)
    print('Train Score: %.2f R2' % (trainScore3))
    testScore3 = r2_score(y_test, y_test_pred)
    print('Test Score: %.2f R2' % (testScore3))

    trainScore4 = mean_absolute_percentage_error(y_train, y_train_pred)
    print('Train Score: %.2f MAPE' % (trainScore4))
    testScore4 = mean_absolute_percentage_error(y_test, y_test_pred)
    print('Test Score: %.2f MAPE' % (testScore4))