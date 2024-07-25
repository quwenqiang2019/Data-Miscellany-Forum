import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)


def base_model(x_train, x_test, y_train, y_test):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)

    sns.set_style('darkgrid')
    font1 = {'family': ['SimSun','Times New Roman'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    # 验证集预测值与真实值的对比
    plt.plot(list(range(0, len(X_test))), y_test, marker='o')
    plt.plot(list(range(0, len(X_test))), y_pred_test, marker='*')
    plt.legend(['真实值', '预测值'])
    plt.xlabel('序列')
    plt.ylabel('房价')
    plt.title('验证集预测值与真实值的对比')
    plt.show()

    # 评价
    df = pd.DataFrame({
                       'explained_variance_score': [round(explained_variance_score(y_test, y_pred_test), 2)],
                        'MAE': [round(mean_absolute_error(y_test, y_pred_test), 2)],
                       'MAPE': [round(mean_absolute_percentage_error(y_test, y_pred_test), 2)],
                        'RMSE': [round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)],
                       'R² score': [round(r2_score(y_test, y_pred_test), 2)]})

    print(df)

    return y_test, y_pred_test


def best_model(x_train, x_test, y_train, y_test):

    model = GradientBoostingRegressor()
    # 定义超参数网格
    param_grid_1 = {'n_estimators':range(1,200,10)}
    param_grid_2 = {'min_samples_split': range(2, 50, 5)}
    param_grid_3 = {'max_depth': range(1, 50, 5)}
    # param_grid_4 = {'min_samples_leaf':range(1,10,1)}
    # param_grid_5 = {'max_features': range(1, 8, 1)}
    param_grid_6 = {'learning_rate': (0.5, 0.1, 0.05, 0.01, 0.005, 0.0001)}
    # param_grid_list = [param_grid_1, param_grid_2, param_grid_3, param_grid_4, param_grid_5, param_grid_6]
    param_grid_list = [param_grid_1, param_grid_2, param_grid_3, param_grid_6]

    # 依次寻找每个参数的最优值
    for param_grid in param_grid_list:
        # 使用GridSearchCV进行超参数搜索
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(x_train, y_train)
        # 输出最佳参数组合和对应的得分
        print("最佳参数组合: ", grid_search.best_params_)
        print("最佳得分: ", grid_search.best_score_)
        print("最佳模型: ", grid_search.best_estimator_)

        cvres = grid_search.cv_results_
        mean_test_score_list = []
        param_list = []
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
            mean_test_score_list.append(np.sqrt(-mean_score))
            param_list.append(params[list(param_grid.keys())[0]])

        sns.set_style('darkgrid')
        font1 = {'family': ['SimSun', 'Times New Roman'], 'weight': 'normal', 'size': 14}
        plt.rc('font', **font1)
        plt.rcParams["axes.unicode_minus"] = False
        plt.plot(param_list, mean_test_score_list, 'o-')
        plt.xlabel(list(param_grid.keys())[0])
        plt.ylabel('mean_test_score')
        plt.show()


    # 一次性选择多个参数的最优组合
    param_grid = {'n_estimators':range(1,200,50),
                  'min_samples_split': range(2, 50, 20),
                  'max_depth': range(1, 50, 20),
                  # 'min_samples_leaf':range(1,10,1),
                  # 'max_features': range(1, 8, 1),
                  'learning_rate': (0.05, 0.01, 0.005)
                  }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(x_train, y_train)

    # 输出最佳参数组合和对应的得分
    print("最佳参数组合: ", grid_search.best_params_)
    print("最佳得分: ", grid_search.best_score_)
    final_model = grid_search.best_estimator_
    print("最佳模型: ", final_model)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    y_pred_test = final_model.predict(x_test)

    sns.set_style('darkgrid')
    font1 = {'family': ['SimSun','Times New Roman'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    # 验证集预测值与真实值的对比
    plt.plot(list(range(0, len(X_test))), y_test, marker='o')
    plt.plot(list(range(0, len(X_test))), y_pred_test, marker='*')
    plt.legend(['真实值', '预测值'])
    plt.xlabel('序列')
    plt.ylabel('房价')
    plt.title('验证集预测值与真实值的对比')
    plt.show()


    # 评价
    df = pd.DataFrame({
                       'explained_variance_score': [round(explained_variance_score(y_test, y_pred_test), 2)],
                        'MAE': [round(mean_absolute_error(y_test, y_pred_test), 2)],
                       'MAPE': [round(mean_absolute_percentage_error(y_test, y_pred_test), 2)],
                        'RMSE': [round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)],
                       'R² score': [round(r2_score(y_test, y_pred_test), 2)]})

    print(df)

    return y_test, y_pred_test




if __name__ == '__main__':
    # 导入数据
    filename = 'data.csv'
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
             'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
    dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
    print(dataset)
    df = pd.DataFrame(dataset)

    #  划分数据集
    features = names[:-1]
    target = ['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


    y_test_base_gb, y_pred_base_gb = base_model(X_train, X_test, y_train, y_test)
    y_test_gs_gb, y_pred_gs_gb = best_model(X_train, X_test, y_train, y_test)
