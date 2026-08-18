# 导入需要的库
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def Read_data(file):
    dt = pd.read_csv(file)
    dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
                  'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina',
                  'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    data =dt
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    print(data.head())
    return data

    # ===================数据清洗======================
def data_clean(data):
    # 重复值处理
    print('存在' if any(data.duplicated()) else '不存在', '重复观测值')
    data.drop_duplicates()

    # 缺失值处理
    # print(data.isnull())
    # print(data.isnull().sum())   #检测每列中缺失值的数量
    # print(data.isnull().T.sum())    #检测每行缺失值的数量
    print('不存在' if any(data.isnull()) else '存在', '缺失值')
    data.dropna()  # 直接删除记录
    data.fillna(method='ffill')  # 前向填充
    data.fillna(method='bfill')  # 后向填充
    data.fillna(value=2)  # 值填充
    data.fillna(value={'resting_blood_pressure': data['resting_blood_pressure'].mean()})  # 统计值填充

    # 异常值处理
    data1 = data['resting_blood_pressure']
    # 标准差监测
    xmean = data1.mean()
    xstd = data1.std()
    print('存在' if any(data1 > xmean + 2 * xstd) else '不存在', '上限异常值')
    print('存在' if any(data1 < xmean - 2 * xstd) else '不存在', '下限异常值')
    # 箱线图监测
    q1 = data1.quantile(0.25)
    q3 = data1.quantile(0.75)
    up = q3 + 1.5 * (q3 - q1)
    dw = q1 - 1.5 * (q3 - q1)
    print('存在' if any(data1 > up) else '不存在', '上限异常值')
    print('存在' if any(data1 < dw) else '不存在', '下限异常值')
    data1[data1 > up] = data1[data1 < up].max()
    data1[data1 < dw] = data1[data1 > dw].min()
    return data


    #===========数值型变量分段统计.离散型变量分组统计==============
def Segment_statistics(data):
    age = data[["age"]]
    bins = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    age2 = pd.cut(age.values.flatten(), bins=bins)
    # print(age2.value_counts())
    age2 = pd.DataFrame(age2, columns=["年龄段"])  #
    age3 = pd.concat([age, age2], axis=1)
    # print(age3)

    tmp3 = data.groupby(['chest_pain_type', 'sex'])
    print(tmp3.count())
    return


    #========================数据编码===========================
def data_encoding(data):
    data = data[["age", 'sex', "chest_pain_type", "resting_blood_pressure", "cholesterol",
                 "fasting_blood_sugar", "rest_ecg","max_heart_rate_achieved", "exercise_induced_angina",
                 "st_depression", "st_slope", "num_major_vessels","thalassemia","target"]]
    Discretefeature=['sex',"chest_pain_type", "fasting_blood_sugar", "rest_ecg",
          "exercise_induced_angina",  "st_slope", "thalassemia"]
    Continuousfeature=["age", "resting_blood_pressure", "cholesterol",
                       "max_heart_rate_achieved","st_depression","num_major_vessels"]

    df = pd.get_dummies(data,columns=Discretefeature)

    df[Continuousfeature]=(df[Continuousfeature]-df[Continuousfeature].mean())/(df[Continuousfeature].std())
    df["target"]=data[["target"]]
    return df

def PCA_analysis(data):
    # X提取变量特征；Y提取目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    pca = PCA(n_components=2)

    reduced_x = pca.fit_transform(X)  # 得到了pca降到2维的数据

    yes_x, yes_y = [], []
    no_x, no_y = [], []

    for i in range(len(reduced_x)):
        if y[i] == 1:
            yes_x.append(reduced_x[i][0])
            yes_y.append(reduced_x[i][1])
        elif y[i] == 0:
            no_x.append(reduced_x[i][0])
            no_y.append(reduced_x[i][1])

    font = {'family': 'Times New Roman',
            'size': 16,
            }
    sns.set(font_scale=1.2)

    plt.rc('font',family='Times New Roman')
    plt.scatter(yes_x, yes_y, c='r', marker='o',label='Yes')
    plt.scatter(no_x, no_y, c='b', marker='x',label='No')
    plt.title("PCA analysis")  # 显示标题
    plt.legend()
    plt.show()
    print(pca.explained_variance_ratio_)  # 输出贡献率


def data_partition(data):
    #======================数据集划分==========================
    # 1.4查看样本是否平衡
    print(data["target"].value_counts())
    # X提取变量特征；Y提取目标变量
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=10)
    feature=list(X.columns)
    return X_train, y_train, X_test, y_test,feature


def Draw_ROC(list1,list2):
    fpr_model,tpr_model,thresholds=roc_curve(list1,list2,pos_label=1)
    roc_auc_model=auc(fpr_model,tpr_model)

    font = {'family': 'Times New Roman',
            'size': 12,
            }
    sns.set(font_scale=1.2)
    plt.rc('font',family='Times New Roman')

    plt.plot(fpr_model,tpr_model,'blue',label='AUC = %0.2f'% roc_auc_model)
    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)
    plt.show()
    return

    #===================================决策树========================================
def DT(X_train, y_train, X_test, y_test,feature):
    tree1 = DecisionTreeClassifier(max_depth=5, random_state=0)
    tree1.fit(X_train, y_train)
    print("\nFinally results of decision tree fitting:")
    print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, y_train)))
    print("Accuract on test set: {:.3f}".format(tree1.score(X_test, y_test)))

    predict_target=tree1.predict(X_test)
    predict_target_prob=tree1.predict_proba(X_test)  # 输出分类概率
    predict_target_prob_dt=predict_target_prob[:,1]

    df = pd.DataFrame({'prob':predict_target_prob_dt,'target':predict_target, 'labels':list(y_test)})

    print('预测正确的个数:',sum(predict_target==y_test))
    print('DT验证集报告：')
    print(metrics.classification_report(y_test,predict_target))      #打印出分类器报告
    print(metrics.confusion_matrix(y_test, predict_target))            #打印混淆矩阵

    print('DT训练集报告：')
    predict_Target=tree1.predict(X_train)
    print(metrics.classification_report(y_train,predict_Target))
    print(metrics.confusion_matrix(y_train, predict_Target))

    id=np.argwhere(tree1.feature_importances_>0)                     #找出重要性大于0的索引

    id=[i for item in id for i in item]                   #二维数组(列表)转化为一维  列表推导式
    dic={}
    for i in id:
        dic.update({feature[i]:tree1.feature_importances_[i]})

    df=pd.DataFrame.from_dict(dic,orient='index',columns=['权重'])
    df=df.reset_index().rename(columns={'index':'特征'})
    df=df.sort_values(by='权重',ascending=False)

    data_hight=df['权重'].values.tolist()
    data_x=df['特征'].values.tolist()

    font = {'family': 'Times New Roman',
            'size': 7,
            }
    sns.set(font_scale=1.2)
    plt.rc('font',family='Times New Roman')

    plt.figure()
    plt.barh(range(len(data_x)), data_hight, color='#6699CC')
    plt.yticks(range(len(data_x)),data_x,fontsize=12)

    plt.tick_params(labelsize=12) #刻度字体大小13
    plt.xlabel('Feature importance',fontsize=14)
    plt.title("DT feature importance analysis",fontsize =14)
    plt.show()
    return list(y_test),list(predict_target_prob_dt)


if __name__=="__main__":
    data1=Read_data("F:\数据杂坛\\0504\heartdisease\Heart-Disease-Data-Set-main\\UCI Heart Disease Dataset.csv")
    data1=data_clean(data1)
    # Segment_statistics(data1)
    data2=data_encoding(data1)
    PCA_analysis(data2)
    X_train, y_train, X_test, y_test,feature= data_partition(data2)

    y_test,predict_target_prob_dt=DT(X_train, y_train, X_test, y_test,feature)
    Draw_ROC(y_test,predict_target_prob_dt)
