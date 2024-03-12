import os
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def calibration_plot(true, pred, n, type):
    """
    参数说明：
    true: 实际标签值
    pred: 模型输出的预测概率
    n: 分组数目 (校准区间中有几个点)
        先加工绘图需要的数据形式：df_cal_trans
        然后绘图，可以选择是否带误差棒
    """
    df_cal = pd.DataFrame({'y_true': true, 'y_pred': pred})  # 现将实际值和预测值拼接成一个dataframe
    print(df_cal)
    df_cal = df_cal.sort_values(by='y_pred')  ## 根据预测概率值进行排序
    df_cal['group'], cut_bin = pd.qcut(df_cal['y_pred'], q=n, retbins=True, labels=list(range(1, n + 1)))  ## 将数据进行分箱
    output_list = list()
    print(df_cal)
    # print(df_cal.loc[df_cal['group'] == i, 'y_true'])
    for i in range(1, n + 1):
        print(i)
        temp = df_cal.loc[df_cal['group'] == i, 'y_true']
        print(temp)
        print(temp.value_counts(1))
        print(temp.value_counts(1)[0])
        true_pos_rate = 1 - df_cal.loc[df_cal['group'] == i, 'y_true'].value_counts(1)[0]
        y_pred_mean = df_cal.loc[df_cal['group'] == i, 'y_pred'].mean()
        y_pred_sd = df_cal.loc[df_cal['group'] == i, 'y_pred'].std()
        output = {'group': i, 'true_pos_rate': true_pos_rate, 'y_pred_mean': y_pred_mean, 'y_pred_sd': y_pred_sd}
        output_list.append(output)
    df_cal_trans = pd.DataFrame(output_list)
    calibration_slop = round(scipy.stats.linregress(df_cal_trans['y_pred_mean'], df_cal_trans['true_pos_rate']).slope,
                             3)

    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.spines.right'] = False  # 不绘制右边的框线
    plt.rcParams['axes.spines.top'] = False  # 不绘制上方的框线
    line = plt.errorbar(df_cal_trans['y_pred_mean'], df_cal_trans['true_pos_rate'],
                        # yerr=df_cal_trans['y_pred_sd'],
                        fmt='--o',  # 数据点标记式样和数据点标记的连线式样
                        ecolor="#00688B",  # 误差棒的颜色
                        elinewidth=0.8,  # 误差棒线条粗细
                        ms=4,  # 数据点大小
                        mfc="#00688B",  # 数据点颜色
                        capthick=1,  # 误差棒边界横线的厚度
                        capsize=2  # 误差棒边界横线的大小
                        )
    limits = round(max(df_cal_trans['true_pos_rate'].max(), df_cal_trans['y_pred_mean'].max()) + 0.02, 3)
    plt.plot([0, limits], [0, limits], "--", lw=1, color="grey")
    plt.xlim(0, limits)
    plt.ylim(0, limits)
    plt.xlabel('Predicted event probability', fontsize=10)
    plt.ylabel('Observed event probability', fontsize=10)
    # plt.legend(handles=[line],labels=['HL P-value: > 0.05'], loc='best')
    plt.legend(handles=[line], labels=['Calibration slope: {}'.format(calibration_slop)], loc='best')  # 'lower right'
    plt.grid(axis="y")  # 设置横向网格线
    plt.savefig(os.path.join(base_dir, 'result', f'calibration{type}.jpg'), bbox_inches='tight', dpi=600)
    plt.show()





def feature_importance(model):
    # 提取特征重要性
    feature_importance = model.feature_importances_
    feature_names = features

    # 创建特征重要性的DataFrame
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    # 对特征重要性进行排序
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df.to_excel(os.path.join(base_dir, 'result', 'rf特征重要性排序.xlsx'))
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance', fontdict={'fontsize': 14})
    plt.xlabel('Importance', fontdict={'fontsize': 14})
    plt.ylabel('Feature', fontdict={'fontsize': 14})
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'result', 'rf.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


def model_evalution_test(model, X_test, y_test):
    # 模型推理与评价
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred) # 准确率acc
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.savefig(os.path.join(base_dir, 'result', 'ConfusionMatrix.jpg'), bbox_inches='tight', dpi=600)

    cr = classification_report(y_test, y_pred) # 分类报告
    print('测试集分类报告\n', cr)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
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
    plt.savefig(os.path.join(base_dir, 'result', 'roc_test.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


    calibration_plot(y_test, y_scores[:, 1], 4, 'test')


def model_evalution_train(model, X_train, y_train):
    # 模型推理与评价
    y_pred = model.predict(X_train)
    y_scores = model.predict_proba(X_train)
    acc = accuracy_score(y_train, y_pred) # 准确率acc
    cm = confusion_matrix(y_train, y_pred) # 混淆矩阵
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.savefig(os.path.join(base_dir, 'result', 'ConfusionMatrix.jpg'), bbox_inches='tight', dpi=600)

    cr = classification_report(y_train, y_pred) # 分类报告
    print('测试集分类报告\n', cr)
    fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
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
    plt.savefig(os.path.join(base_dir, 'result', 'roc_train.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


    calibration_plot(y_train, y_scores[:, 1], 2, 'train')




if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    # 准备数据
    data = pd.read_excel(os.path.join(base_dir, 'data', '随机森林数据.xlsx'))
    df = pd.DataFrame(data)
    df = df.dropna()  # 直接删除记录

    # 目标变量和特征变量
    target = '结局变量'
    features = df.columns.drop(target)
    print(df[target].value_counts())  # 顺便查看一下样本是否平衡

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # 训练结果
    print("训练集准确率:", model.score(X_train, y_train))
    print("测试集准确率:", model.score(X_test, y_test))
    model_evalution_train(model, X_train, y_train)
    model_evalution_test(model, X_test, y_test)

    # 特征重要性
    feature_importance(model)










