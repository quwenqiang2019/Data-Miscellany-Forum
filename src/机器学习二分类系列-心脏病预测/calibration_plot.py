import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def calibration_plot(true ,pred ,n):
    """
    参数说明：
    true: 实际标签值
    pred: 模型输出的预测概率
    n: 分组数目 (校准区间中有几个点)
        先加工绘图需要的数据形式：df_cal_trans
        然后绘图，可以选择是否带误差棒
    """
    df_cal = pd.DataFrame({'y_true' :true ,'y_pred' :pred}) # 现将实际值和预测值拼接成一个dataframe
    df_cal = df_cal.sort_values(by='y_pred') ## 根据预测概率值进行排序
    df_cal['group'], cut_bin = pd.qcut(df_cal['y_pred'] ,q=n ,retbins=True ,labels = list(range(1 , n +1))) ## 将数据进行分箱
    output_list = list()
    for i in range(1 , n +1):
        true_pos_rate = 1 - df_cal.loc[df_cal['group' ]==i ,'y_true'].value_counts(1)[0]
        y_pred_mean = df_cal.loc[df_cal['group' ]==i ,'y_pred'].mean()
        y_pred_sd = df_cal.loc[df_cal['group' ]==i ,'y_pred'].std()
        output = {'group' :i ,'true_pos_rate' :true_pos_rate ,'y_pred_mean' :y_pred_mean ,'y_pred_sd' :y_pred_sd}
        output_list.append(output)
    df_cal_trans = pd.DataFrame(output_list)
    calibration_slop = round(scipy.stats.linregress(df_cal_trans['y_pred_mean'] ,df_cal_trans['true_pos_rate']).slope
                             ,3)

    plt.figure(figsize=(6 ,4))
    plt.rcParams['axes.spines.right'] = False  # 不绘制右边的框线
    plt.rcParams['axes.spines.top'] = False    # 不绘制上方的框线
    line = plt.errorbar(df_cal_trans['y_pred_mean'] ,df_cal_trans['true_pos_rate'],
                        # yerr=df_cal_trans['y_pred_sd'],
                        fmt='--o', # 数据点标记式样和数据点标记的连线式样
                        ecolor="#00688B", # 误差棒的颜色
                        elinewidth=0.8,  # 误差棒线条粗细
                        ms=4, # 数据点大小
                        mfc = "#00688B", # 数据点颜色
                        capthick = 1, # 误差棒边界横线的厚度
                        capsize = 2  # 误差棒边界横线的大小
                        )
    limits = round(max(df_cal_trans['true_pos_rate'].max() ,df_cal_trans['y_pred_mean'].max()) + 0.02 ,3)
    plt.plot([0 ,limits] ,[0 ,limits] ,"--" ,lw=1 ,color="grey")
    plt.xlim(0 ,limits)
    plt.ylim(0 ,limits)
    plt.xlabel('Predicted event probability' ,fontsize=10)
    plt.ylabel('Observed event probability' ,fontsize=10)
    # plt.legend(handles=[line],labels=['HL P-value: > 0.05'], loc='best')
    plt.legend(handles=[line] ,labels=['Calibration slope: {}'.format(calibration_slop)], loc='best') # 'lower right'
    plt.grid(axis="y") # 设置横向网格线
    plt.show()
    # return df_cal_trans


if __name__ == '__main__':
    # 准备数据
    data = pd.read_csv(r'Dataset.csv')
    df = pd.DataFrame(data)

    # 提取目标变量和特征变量
    target = 'target'
    features = df.columns.drop(target)
    print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)


    # 归一化
    mm1 = MinMaxScaler()   # 特征进行归一化
    X_train_m = mm1.fit_transform(X_train)
    mm2 = MinMaxScaler()     # 标签进行归一化
    y_train_m = mm2.fit_transform(y_train)


    # 模型的构建与训练
    model = LogisticRegression()
    model.fit(X_train_m, y_train_m)

    # 模型推理与评价
    # 对测试集特征进行相同规则mm1的归一化处理，然后输入到模型进行预测
    X_test_m = mm1.transform(X_test) #注意fit_transform() 和 transform()的区别
    y_pred_m = model.predict(X_test_m) #利用输入特征input1和input2测试模型
    y_scores = model.predict_proba(X_test_m)
    y_pred = mm2.inverse_transform(np.reshape(y_pred_m, (-1, 1)))

    calibration_plot(y_test[target], list(y_scores[:, 1]), 3)