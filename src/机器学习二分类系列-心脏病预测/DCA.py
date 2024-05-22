import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax


if __name__ == '__main__':
    # 准备数据
    data = pd.read_csv(r'Dataset.csv')
    df = pd.DataFrame(data)

    # 提取目标变量和特征变量
    target = 'target'
    features = df.columns.drop(target)
    print(data["target"].value_counts())  # 顺便查看一下样本是否平衡

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)

    # 归一化
    mm1 = MinMaxScaler()  # 特征进行归一化
    X_train_m = mm1.fit_transform(X_train)
    mm2 = MinMaxScaler()  # 标签进行归一化
    y_train_m = mm2.fit_transform(y_train)

    # 模型的构建与训练
    model = LogisticRegression()
    model.fit(X_train_m, y_train_m)

    # 模型推理与评价
    # 对测试集特征进行相同规则mm1的归一化处理，然后输入到模型进行预测
    X_test_m = mm1.transform(X_test)  # 注意fit_transform() 和 transform()的区别
    y_pred_m = model.predict(X_test_m)  # 利用输入特征input1和input2测试模型
    y_scores = model.predict_proba(X_test_m)
    y_pred = mm2.inverse_transform(np.reshape(y_pred_m, (-1, 1)))

    thresh_group = np.arange(0, 1, 0.05)
    net_benefit_model = calculate_net_benefit_model(thresh_group, list(y_scores[:, 1]), y_test)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
    # fig.savefig('fig1.png', dpi = 300)
    plt.show()
