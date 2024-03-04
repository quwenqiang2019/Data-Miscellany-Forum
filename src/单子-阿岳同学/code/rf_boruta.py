import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy


# 准备数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'feature selection.csv'))
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'Landcover'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)
print(type(X_train.values))
print(y_train.values)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=0)
feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X_train.values, y_train.values)

# 原始特征
print('\n Initial features: ', features)#打印训练数据集中初始特征的列名，不包括’id,target‘
# boruta选择的特征个数
print('\n Number of selected features:')
print(feat_selector.n_features_)#运行Boruta后的所选特征数量
# top特征（也就是被选择的特征）
feature_df = pd.DataFrame(features, columns=['features'])
feature_df['rank']=feat_selector.ranking_
feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)#把特征重要性排序，并重置索引，新排名存储在feature_df中
print('\n Top %d features:' % feat_selector.n_features_)
print(feature_df.head(feat_selector.n_features_))
feature_df.to_csv(os.path.join(base_dir, 'result', 'rf-boruta-feature-ranking.csv'))#把特征排名保存到数据帧里

# 原始特征的排序
print('\n Feature ranking:')
print(feat_selector.ranking_)#输出各个特征的重要性排名

# 选择的特征
print('\n Selected features:')
print(feat_selector.support_)#返回一个布尔类型的数组，代表每个特征是否被选择。如果特征被选择，对应的值为True，否则为False。
print('\n Support for weak features:')
print(feat_selector.support_weak_)#返回一个布尔类型的数组，代表是否存在一些弱特征被选择。如果特征是弱特征且被选择，对应的值为True，否则为False。

