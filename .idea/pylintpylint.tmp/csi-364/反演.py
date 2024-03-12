import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
import tensorflow as tf
import os
from joblib import dump
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)

def read_data(filename):
    dataset = pd.read_excel(os.path.join(BASE_DIR, 'data', filename))
    dataset = pd.DataFrame(dataset)
    dataset = dataset.replace('—', np.nan)
    dataset.fillna(dataset.mean(), inplace=True)
    new_column_names = {'氨氮\n（mg/L）': '氨氮',
                        '硝酸盐氮\n（mg/L）': '硝酸盐氮',
                        '总氮\n（mg/L）': '总氮',
                        '总磷\n（mg/L）': '总磷',
                        'CODcr\n（mg/L）': 'CODcr',
                        '高锰酸钾指数\n（mg/L）': '高锰酸钾指数',
                        '悬浮物\n（mg/L）': '悬浮物',
                        '浊度\n（NTU）': '浊度',
                        '叶绿素a\n（ug/L）': '叶绿素a'}
    dataset = dataset.rename(columns=new_column_names)

    print(dataset.head())
    dataset_parameter = ['FAI', '氨氮', '硝酸盐氮', '总氮', '总磷', 'CODcr', '高锰酸钾指数', '悬浮物', '浊度', '叶绿素a','pH']

    FAI = np.array(dataset[dataset_parameter[0]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    NH3N = np.array(dataset[dataset_parameter[1]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    NO3N = np.array(dataset[dataset_parameter[2]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    TN = np.array(dataset[dataset_parameter[3]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    TP = np.array(dataset[dataset_parameter[4]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    CODcr = np.array(dataset[dataset_parameter[5]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    PPI = np.array(dataset[dataset_parameter[6]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    TSM = np.array(dataset[dataset_parameter[7]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    Tb = np.array(dataset[dataset_parameter[8]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    Chla = np.array(dataset[dataset_parameter[9]]).reshape(len(dataset[dataset_parameter[0]]), 1)
    PH = np.array(dataset[dataset_parameter[10]]).reshape(len(dataset[dataset_parameter[0]]), 1)


    dataset_band = ['B1','B2','B3','B4','B5','B6','B7','B8','B8a']
    B1 = np.array(dataset[dataset_band[0]])
    B2 = np.array(dataset[dataset_band[1]])
    B3 = np.array(dataset[dataset_band[2]])
    B4 = np.array(dataset[dataset_band[3]])
    B5 = np.array(dataset[dataset_band[4]])
    B6 = np.array(dataset[dataset_band[5]])
    B7 = np.array(dataset[dataset_band[6]])
    B8 = np.array(dataset[dataset_band[7]])
    B8a = np.array(dataset[dataset_band[8]])

    return FAI, NH3N, NO3N, TN, TP, CODcr, PPI, TSM, Tb, Chla, PH, B1, B2, B3, B4, B5, B6, B7, B8, B8a



def kf(model, X_data, Y_data):
    # ====================利用交叉验证选择最佳模型数据集划分=================
    kf = KFold(n_splits=4, shuffle=True, random_state=8)
    score = 0
    best_score = -float('inf')  # 初始化为正无穷大
    best_train_index = None
    best_test_index = None
    for train_index, test_index in kf.split(X_data):
        x_train = X_data[train_index]
        y_train = Y_data[train_index]
        x_test = X_data[test_index]
        y_test = Y_data[test_index]
        clt = model.fit(x_train, y_train)
        curr_score = clt.score(x_test, y_test)
        # print("准确率为：", curr_score)
        score = score + curr_score

        # 如果当前数据集划分的性能更好，则更新最佳数据集划分
        if curr_score > best_score:
            best_score = curr_score
            best_train_index = train_index
            best_test_index = test_index

    x_train = X_data[best_train_index]
    y_train = Y_data[best_train_index]
    x_test = X_data[best_test_index]
    y_test = Y_data[best_test_index]

    return x_train, x_test, y_train, y_test


def log(B, parameter, B_name, parameter_name,  flag):

    X_data = B
    Y_data = parameter
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=1 / 4, random_state=8)


    if flag == 0:
        # 拟合y = a*ln(x) + b形式的对数函数
        a1, b = np.polyfit(x_train.flatten(), y_train.flatten(), 1, w=np.log(x_train.flatten()))
        f = np.poly1d([a1, b])
        print("y=%.4f*x+%.4f" % (a1, b))
        y_pred_test = f(x_test.flatten())         # 用训练后的模型，进行预测


    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False


    # 简单画图显示1
    plt.scatter(list(np.array(x_train).flatten()), list(np.array(y_train).flatten()), s=50,  c="red")
    # 计算拟合直线的斜率和截距
    slope, intercept = np.polyfit(list(np.array(x_train).flatten()), list(np.array(y_train).flatten()), 1)
    # 保留小数点后三位
    slope = round(slope, 3)
    intercept = round(intercept, 3)
    plt.plot(np.array(x_train).flatten(), slope*(np.array(x_train).flatten()) + intercept, c="blue", label=f"y = {slope}*x + {intercept}")
    plt.title(rf"{parameter_name}-{B_name}拟合", fontproperties=font1)
    plt.xlabel(rf'{B_name}', fontproperties=font1)
    plt.ylabel(rf'{parameter_name}', fontproperties=font1)
    plt.legend(fontsize=14)
    save_folder = os.path.join(BASE_DIR, rf'{parameter_name}', 'result')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.savefig(os.path.join(save_folder, rf'{parameter_name}-{B_name}对数训练集反演.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 简单画图显示2
    plt.scatter(list(np.array(y_test).flatten()), list(np.array(y_pred_test).flatten()), s=50, c="red", label='测试样本点')
    x2 = np.arange(np.min(y_test), np.max(y_test), 0.1)
    y2 = x2
    plt.plot(x2, y2, "blue", label='y = x')
    plt.title(rf"{parameter_name}-{B_name}验证", fontproperties=font1)
    plt.xlabel('测量值', fontproperties=font1)
    plt.ylabel('预测值', fontproperties=font1)
    plt.legend(fontsize=14)
    save_folder = os.path.join(BASE_DIR, rf'{parameter_name}', 'result')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, rf'{parameter_name}-{B_name}对数测试集验证.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    df = pd.DataFrame({'拟合方程': ["y=%.4f*x+%.4f" % (a1, b)], "拟合优度": [r2_score(y_test, y_pred_test)],
                       'explained_variance_score': [round(explained_variance_score(y_test, y_pred_test), 2)], 'MAE': [round(mean_absolute_error(y_test, y_pred_test), 2)],
                       'MAPE': [round(mean_absolute_percentage_error(y_test, y_pred_test), 2)], 'RMSE': [round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)],
                       'R² score': [round(r2_score(y_test, y_pred_test), 2)]})

    print(df)

    df.to_excel(writer, sheet_name=f'对数{B_name}', index=False)

    return y_pred_test


def lr(B, parameter, B_name, parameter_name,  flag):
    '''
    线性回归拟合，单
    '''

    X_data = B
    Y_data = parameter
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=1/4, random_state=8)
    model = LinearRegression()
    # x_train, x_test, y_train, y_test = kf(model, X_data, Y_data)

# =======================================================
    if flag == 0:
        model.fit(x_train, y_train)
        a1 = round(model.coef_[0][0], 2) # coef_是系数，intercept_是截距
        b = round(model.intercept_[0], 2)
        print("y=%.4f*x+%.4f" % (a1, b))
        print("得分", model.score(x_train, y_train))  # 决定系数（coefficient ofdetermination），有的教材上翻译为判定系数，也称为拟合优度。
        y_pred_test = model.predict(x_test)         # 用训练后的模型，进行预测

    elif flag == 1:        # 数据归一化
        mm1 = MinMaxScaler()  # 特征进行归一化
        x_train_scaled = mm1.fit_transform(x_train)
        mm2 = MinMaxScaler()  # 标签进行归一化
        y_train_scaled = mm2.fit_transform(y_train)

        model.fit(x_train_scaled, y_train_scaled)        # 训练模型
        a1 = round(model.coef_[0][0], 2) # coef_是系数，intercept_是截距
        b = round(model.intercept_[0], 2)
        print("y=%.4f*x+%.4f" % (a1, b))
        print("得分", model.score(x_train, y_train))  # 决定系数（coefficient ofdetermination），有的教材上翻译为判定系数，也称为拟合优度。
        x_test_scaled = mm1.transform(x_test)# 用训练后的模型，进行预测
        y_pred_test_scaled = model.predict(x_test_scaled)
        y_pred_test = mm2.inverse_transform(y_pred_test_scaled)

    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False

    # 简单画图显示1
    plt.scatter(list(np.array(x_train).flatten()), list(np.array(y_train).flatten()), s=50,  c="red")
    # 计算拟合直线的斜率和截距
    slope, intercept = np.polyfit(list(np.array(x_train).flatten()), list(np.array(y_train).flatten()), 1)
    # 保留小数点后三位
    slope = round(slope, 3)
    intercept = round(intercept, 3)
    plt.plot(np.array(x_train).flatten(), slope*(np.array(x_train).flatten()) + intercept, c="blue", label=f"y = {slope}*x + {intercept}")
    plt.title(rf"{parameter_name}-{B_name}拟合", fontproperties=font1)
    plt.xlabel(rf'{B_name}', fontproperties=font1)
    plt.ylabel(rf'{parameter_name}', fontproperties=font1)
    plt.legend(fontsize=14)
    save_folder = os.path.join(BASE_DIR, rf'{parameter_name}', 'result')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.savefig(os.path.join(save_folder, rf'{parameter_name}-{B_name}训练集反演.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 简单画图显示2
    plt.scatter(list(np.array(y_test).flatten()), list(np.array(y_pred_test).flatten()), s=50, c="red", label='测试样本点')
    x2 = np.arange(np.min(y_test), np.max(y_test), 0.1)
    y2 = x2
    plt.plot(x2, y2, "blue", label='y = x')
    plt.title(rf"{parameter_name}-{B_name}验证", fontproperties=font1)
    plt.xlabel('测量值', fontproperties=font1)
    plt.ylabel('预测值', fontproperties=font1)
    plt.legend(fontsize=14)
    save_folder = os.path.join(BASE_DIR, rf'{parameter_name}', 'result')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, rf'{parameter_name}-{B_name}测试集验证.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    df = pd.DataFrame({'拟合方程': ["y=%.4f*x+%.4f" % (a1, b)], "得分": [model.score(x_train, y_train)],
                       'explained_variance_score': [round(explained_variance_score(y_test, y_pred_test), 2)], 'MAE': [round(mean_absolute_error(y_test, y_pred_test), 2)],
                       'MAPE': [round(mean_absolute_percentage_error(y_test, y_pred_test), 2)], 'RMSE': [round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)],
                       'R² score': [round(r2_score(y_test, y_pred_test), 2)]})

    print(df)

    df.to_excel(writer, sheet_name=B_name, index=False)

    return y_pred_test



def ml(B, parameter, parameter_name, model, model_name, flag):
    X_data = B
    Y_data = parameter
    # ====================利用交叉验证选择最佳模型数据集划分=================
    # x_train, x_test, y_train, y_test = kf(model, X_data, Y_data)
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=1 / 4, random_state=8)
    if flag == 0:
        model.fit(x_train, y_train)
        # # 保存模型为 Pickle 文件
        # save_folder = os.path.join(BASE_DIR, 'result', f'{parameter_name}')
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        # dump(model, os.path.join(save_folder, f'{model_name}.pkl'))

        # 模型推理
        y_pred_test = model.predict(x_test)

        # 读文件用，如果直接读文件预测，将上面三行代码注释
        # model_ = joblib.load(os.path.join(BASE_DIR, 'result', 'TSM', 'GB.pkl'))
        # y_pred_test = model_.predict(x_test)

    elif flag == 1:
        # 数据归一化
        mm1 = MinMaxScaler()  # 特征进行归一化
        x_train_scaled = mm1.fit_transform(x_train)
        mm2 = MinMaxScaler()  # 标签进行归一化
        y_train_scaled = mm2.fit_transform(y_train)
        # 训练模型
        model.fit(x_train_scaled, y_train_scaled)
        # 模型推理
        x_test_scaled = mm1.transform(x_test)
        y_pred_test_scaled = model.predict(x_test_scaled)
        y_pred_test = mm2.inverse_transform(y_pred_test_scaled)

    # 绘图风格设置,使用seaborn库的API来设置样式
    sns.set_style('darkgrid')
    font1 = {'family': ['SimSun'], 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False


    print(model_name, list(np.array(y_test).flatten()), list(np.array(y_pred_test).flatten()))
    plt.scatter(list(np.array(y_test).flatten()), list(np.array(y_pred_test).flatten()), s=50, c="red", label='测试点')
    x2 = np.arange(np.min(y_test), np.max(y_test), 0.1)
    y2 = x2

    plt.plot(x2, y2, "blue", label='y = x')
    plt.title(f"{model_name}模型验证", fontproperties=font1)
    plt.xlabel('测量值', fontproperties=font1)
    plt.ylabel('预测值', fontproperties=font1)
    plt.legend(fontsize=14)
    save_folder = os.path.join(BASE_DIR, rf'{parameter_name}', 'result')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, f'{model_name}拟合模型.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    # 评价
    df = pd.DataFrame({"得分": [model.score(x_train, y_train)],
                       'explained_variance_score': [round(explained_variance_score(y_test, y_pred_test), 2)], 'MAE': [round(mean_absolute_error(y_test, y_pred_test), 2)],
                       'MAPE': [round(mean_absolute_percentage_error(y_test, y_pred_test), 2)], 'RMSE': [round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)],
                       'R² score': [round(r2_score(y_test, y_pred_test), 2)]})

    print(df)

    df.to_excel(writer, sheet_name=model_name, index=False)


    return y_pred_test



if __name__=="__main__":
    FAI, NH3N, NO3N, TN, TP, CODcr, PPI, TSM, Tb, Chla, PH, B1, B2, B3, B4, B5, B6, B7, B8, B8a = read_data('dataset.xlsx')

    parameter = Chla
    parameter_name = 'Chla'

    if not os.path.exists(f'result'):
        os.makedirs(f'result')
    writer = pd.ExcelWriter(f'result/{parameter_name}.xlsx')
    # # ================================TSM========================================
    # # 1、单因素回归建模
    index1 = B4
    index2 = B8 / B8a
    index3 = B5 - B4
    index4 = (B8 - B8a) / (B8 + B8a)
    y_pred_test_index1 = lr(B=np.reshape(index1, (-1, 1)), parameter=parameter, B_name='B4', parameter_name=parameter_name, flag=0)
    print('-------------------')
    y_pred_test_index2 = lr(B=np.reshape(index2, (-1, 1)), parameter=parameter, B_name=r'B8比B8a', parameter_name=parameter_name, flag=0)
    print('-------------------')
    y_pred_test_index3 = lr(B=np.reshape(index3, (-1, 1)), parameter=parameter, B_name='B5-B4', parameter_name=parameter_name, flag=0)
    print('-------------------')
    y_pred_test_index4 = lr(B=np.reshape(index4, (-1, 1)), parameter=parameter, B_name='(B8-B8a)比(B8+Ba)', parameter_name=parameter_name, flag=0)
    print('-------------------')




    # 1、对数

    y_pred_test_log_index1 = log(B=np.reshape(index1, (-1, 1)), parameter=parameter, B_name='B4', parameter_name=parameter_name, flag=0)
    print('-------------------')
    y_pred_test_log_index2 = log(B=np.reshape(index2, (-1, 1)), parameter=parameter, B_name=r'B8比B8a', parameter_name=parameter_name, flag=0)
    print('-------------------')
    y_pred_test_log_index3 = log(B=np.reshape(index3, (-1, 1)), parameter=parameter, B_name='B5-B4', parameter_name=parameter_name, flag=0)
    print('-------------------')
    y_pred_test_log_index4 = log(B=np.reshape(index4, (-1, 1)), parameter=parameter, B_name='(B8-B8a)比(B8+Ba)', parameter_name=parameter_name, flag=0)
    print('-------------------')










    # # # # 2、多因素机器学习建模
    composition_index = np.column_stack((index1, index2, index3, index4))
    print('-------------------')
    y_pred_test_lr = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=LinearRegression(), model_name='LR', flag=0)
    print('-------------------')
    y_pred_test_knn = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=KNeighborsRegressor(), model_name='KNN', flag=0)
    print('-------------------')
    y_pred_test_svr = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=SVR(), model_name='SVR', flag=0)
    print('-------------------')
    y_pred_test_dt = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=DecisionTreeRegressor(), model_name='DT', flag=0)
    print('-------------------')
    y_pred_test_rf = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=RandomForestRegressor(), model_name='RF', flag=0)
    print('-------------------')
    y_pred_test_gb = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=GradientBoostingRegressor(), model_name='GB',flag=0)
    print('-------------------')
    y_pred_test_xgb = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=XGBRegressor(), model_name='XGB', flag=0)
    print('-------------------')
    y_pred_test_ = ml(B=composition_index, parameter=parameter, parameter_name=parameter_name, model=MLPRegressor(), model_name='MLP', flag=0)
    print('-------------------')


    writer.save()
