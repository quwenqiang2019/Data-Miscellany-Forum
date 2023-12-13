# 导入需要的库
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import scipy.stats as stats

# =============读取数据===========
def Read_data(file):
    dt = pd.read_csv(file)
    dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar','rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope','num_major_vessels', 'thalassemia', 'target']
    data = dt
    return data

# ===========数据清洗==============
def data_clean(data):
    # 重复值处理
    print('存在' if any(data.duplicated()) else '不存在', '重复观测值')
    data.drop_duplicates()
    # 缺失值处理
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
    print(data)
    return data

#=============mannwhitneyu检验======================
def mannwhitneyu_age(data):
    data1=data[['age','target']]
    print(data1['age'].values)
    print(np.mean(data1['age'].values), np.std(data1['age'].values))
    data_age_Y=data1[(data1['target'] ==1)]
    print(len(data_age_Y))
    print(data_age_Y['age'].values)
    print(np.mean(data_age_Y['age'].values), np.std(data_age_Y['age'].values))
    data_age_N=data1[(data1['target'] ==0)]
    print(len(data_age_N))
    print(data_age_N['age'].values)
    print(np.mean(data_age_N['age'].values), np.std(data_age_N['age'].values))
    MWU=stats.mannwhitneyu(data_age_Y['age'].values, data_age_N['age'].values, alternative='two-sided')
    print(MWU)

#============主函数==============
if __name__=="__main__":
    data1=Read_data("F:\数据杂坛\\0504\heartdisease\Heart-Disease-Data-Set-main\\UCI Heart Disease Dataset.csv")
    data1=data_clean(data1)
    mannwhitneyu_age(data1)
