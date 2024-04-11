import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
import numpy as np

# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df.head())

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 随机森林分类器创建一个Boruta的选择对象
model = RandomForestClassifier(n_estimators=100, random_state=0)
feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(np.array(X_train), np.array(y_train))

# 原始特征的排序
print('\n Feature ranking:')
print(pd.DataFrame({"feature":features, 'feature_ranking':feat_selector.ranking_})) #输出各个特征的重要性排名
print('\n Selected features:')
print(pd.DataFrame({"feature":features, 'Selected':feat_selector.support_}))  #返回一个布尔类型的数组，代表每个特征是否被选择。如果特征被选择，对应的值为True，否则为False。
print('\n Support for weak features:')
print(pd.DataFrame({"feature":features, 'Selected':feat_selector.support_weak_}))  #返回一个布尔类型的数组，代表是否存在一些弱特征被选择。如果特征是弱特征且被选择，对应的值为True，否则为False。
