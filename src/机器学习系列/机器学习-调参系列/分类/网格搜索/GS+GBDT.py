import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)


def base_model(x_train, x_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)

    # 模型推理与评价
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)  # 准确率acc
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    cr = classification_report(y_test, y_pred)  # 分类报告
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1)  # 计算ROC曲线和AUC值,绘制ROC曲线
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return



def best_model(x_train, x_test, y_train, y_test):

    model = GradientBoostingClassifier()
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

    sns.set_style('darkgrid')
    font1 = {'family': ['SimSun','Times New Roman'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False
    # 模型推理与评价
    y_pred = final_model.predict(X_test)
    y_scores = final_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)  # 准确率acc
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    cr = classification_report(y_test, y_pred)  # 分类报告
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1)  # 计算ROC曲线和AUC值,绘制ROC曲线
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()




if __name__ == '__main__':
    # 准备数据
    data = pd.read_csv(r'Dataset.csv')
    df = pd.DataFrame(data)
    ## 数据基本信息
    print(df.head())
    print(df.info())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]  # 类别型变量名
    num_cols = [col for col in df.columns if df[col].dtype != "object"]  # 数值型变量名

    # 提取目标变量和特征变量
    target = 'target'
    features = df.columns.drop(target)
    print(data["target"].value_counts())  # 顺便查看一下样本是否平衡

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

    base_model(X_train, X_test, y_train, y_test)
    best_model(X_train, X_test, y_train, y_test)
