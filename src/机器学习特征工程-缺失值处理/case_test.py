import pandas as pd
import numpy as np

data = pd.DataFrame({
    'name': ['Bob', 'Mary', 'Peter', 'nancy', 'Lucy'],
    'score': [99, 100, np.nan, 91, 95],
    'class': ['class1', 'class2', 'class1', 'class2', np.nan],
    'sex': ['male', 'fmale', 'male', 'male', 'fmale'],
    'age': [23, 25, np.nan, 19, 24]
})

# data[['score', 'age']] = data[['score', 'age']].fillna(data[['score', 'age']].mean())
# data['class'] = data['class'].fillna(data['class'].mode()[0])
#
# print(data)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 获取数据
data_train = data.iloc[[0, 1, 3]]

data_train_x = data_train[['age']]
data_train_y = data_train['class']
print(data_train_x, data_train_y)
# 使用决策树进行拟合
clf = LogisticRegression()
clf.fit(data_train_x, data_train_y)
print(pd.DataFrame(data[['age']].iloc[4]))
print(clf.predict(pd.DataFrame(data[['age']].iloc[4])))

# 使用分类结果进行填充
# data['class'].iloc[4] = clf.predict(pd.DataFrame(data[['age']].iloc[4]))[0]
