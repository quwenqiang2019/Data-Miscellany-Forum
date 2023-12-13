import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,LassoCV
import seaborn as sns
#=================================读取数据============================
class Solution():
    def __init__(self):
        feature = ['男', '女', '年龄', 'CCP-正常', 'CCP-异常', 'MCV-正常', 'MCV-异常',
                   'AKA-正常', 'AKA-异常','RF-正常', 'RF-异常', 'ANA-正常', 'ANA-异常',
                   'ds-DNA-正常', 'ds-DNA-异常','CRP-正常', 'CRP-异常', 'ESR-正常', 'ESR-异常',
                   '尿蛋白-正常', '尿蛋白-异常', '尿潜血-正常', '尿潜血-异常','尿红细胞-正常',
                   '尿红细胞-异常', 'WBC-正常', 'WBC-异常', 'Hb-正常', 'Hb-异常', 'PLT-正常',
                   'PLT-异常', 'ALT-正常', 'ALT-异常', 'AST-正常', 'AST-异常', 'r-GT-正常',
                   'r-GT-异常', 'TBIL-正常', 'TBIL-异常', 'ALB-正常','ALB-异常', 'GLB-正常',
                   'GLB-异常',  'A/O-正常', 'A/O-异常', 'Cr-正常', 'Cr-异常', 'BUN-正常',
                   'BUN-异常', 'UA-正常', 'UA-异常', 'C3-正常', 'C3-异常', 'C4-正常', 'C4-异常',
                   'IgA-正常', 'IgA-异常', 'IgG-正常','IgG-异常', 'IgE-正常', 'IgE-异常',
                   '晨僵正常', '晨僵异常', '发热正常', '发热异常', '雷诺正常', '雷诺异常',
                   '口眼干正常', '口眼干异常', '头晕正常', '头晕异常', '四肢正常', '四肢异常',
                   '胸部CT正常', '胸部CT异常', '肺结节正常', '肺结节异常', '诊断结果']
        self.feature=feature

    def Data_sort(self,file):
        data = pd.read_excel(file)
        data = pd.DataFrame(data)
        random_state_value = 90  # 随机种子
        sample_number = 82  # 欠采样数目
        def norm_2(x):
            return (x - stats['min']) / (stats['max']-stats['min'])
        gy_list=['年龄']
        data_gy=data[gy_list]
        stats = data_gy.describe()
        stats = stats.transpose()
        data[gy_list]=norm_2(data_gy)
        data1 = data[self.feature]
        data1 = data1.dropna()  # 删除含缺失值的行
        data1=data1[~data1['诊断结果'].isin([2])]
        print(len(data1))
        dataset=data1
        train_dataset = dataset.sample(frac=0.7, random_state=random_state_value)
        test_dataset = dataset.drop(train_dataset.index)
        print(len(test_dataset))
        train_dataset[train_dataset['诊断结果'].isin([1])]=\
            train_dataset[train_dataset['诊断结果'].isin([1])].iloc[:sample_number]
        train_NRA=train_dataset[train_dataset['诊断结果'].isin([0])]
        train_RA=train_dataset[train_dataset['诊断结果'].isin([1])]
        train_dataset=train_NRA.append(train_RA)
        train_dataset=train_dataset.sample(frac=1,random_state=0)
        print(len(train_dataset))
        train_labels =train_dataset.pop('诊断结果')
        test_labels =test_dataset.pop('诊断结果')
        return train_dataset,train_labels,test_dataset,test_labels

#=========================Lasso变量筛选=======================
    def optimal_lambda_value(self):
        Lambdas = np.logspace(-5, 2, 200)    #10的-5到10的2次方
        # 构造空列表，用于存储模型的偏回归系数
        lasso_cofficients = []
        for Lambda in Lambdas:
            lasso = Lasso(alpha = Lambda, normalize=True, max_iter=10000)
            lasso.fit(train_dataset, train_labels)
            lasso_cofficients.append(lasso.coef_)
        # 绘制Lambda与回归系数的关系
        plt.plot(Lambdas, lasso_cofficients)
        # 对x轴作对数变换
        plt.xscale('log')
        # 设置折线图x轴和y轴标签
        plt.xlabel('Lambda')
        plt.ylabel('Cofficients')
        # 显示图形
        plt.show()
        # LASSO回归模型的交叉验证
        lasso_cv = LassoCV(alphas = Lambdas, normalize=True, cv = 10, max_iter=10000)
        lasso_cv.fit(train_dataset, train_labels)
        # 输出最佳的lambda值
        lasso_best_alpha = lasso_cv.alpha_
        print(lasso_best_alpha)
        return lasso_best_alpha

    # 基于最佳的lambda值建模
    def model(self,train_dataset, train_labels,lasso_best_alpha):
        lasso = Lasso(alpha = lasso_best_alpha, normalize=True, max_iter=10000)
        lasso.fit(train_dataset, train_labels)
        return lasso

    def feature_importance(self,lasso):
        # 返回LASSO回归的系数
        dic={'特征':train_dataset.columns,'系数':lasso.coef_}
        df=pd.DataFrame(dic)
        df1=df[df['系数']!=0]
        print(df1)
        coef = pd.Series(lasso.coef_, index=train_dataset.columns)
        imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
        sns.set(font_scale=1.2)
        # plt.rc('font', family='Times New Roman')
        plt.rc('font', family='simsun')
        imp_coef.plot(kind="barh")
        plt.title("Lasso回归模型")
        plt.show()
        return df1

    def prediction(self,lasso):
        # lasso_predict = lasso.predict(test_dataset)
        lasso_predict = np.round(lasso.predict(test_dataset))
        print(sum(lasso_predict==test_labels))
        print(metrics.classification_report(test_labels,lasso_predict))
        print(metrics.confusion_matrix(test_labels, lasso_predict))
        RMSE = np.sqrt(mean_squared_error(test_labels,lasso_predict))
        print(RMSE)
        return RMSE

if __name__=="__main__":
    Object1=Solution()
    train_dataset, train_labels, test_dataset, test_labels=\
    Object1.Data_sort('F:\医学大数据课题\RA预测\RA预测\特征.xlsx')
    lasso_best_alpha=Object1.optimal_lambda_value()
    lasso=Object1.model(train_dataset, train_labels,lasso_best_alpha)
    feature_choose=Object1.feature_importance(lasso)
    RMSE=Object1.prediction(lasso)
