from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
# 准备数据
data = pd.read_csv(r'G:\数据杂坛\\UCI Heart Disease Dataset.csv')
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
X = df[features].values
y = df[target].values

# 法1====主要用于找出单个模型最佳的数据集划分
kf = KFold(n_splits = 5, shuffle=True, random_state=0)
score=0
for train_index, test_index in kf.split(y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clt = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
    curr_score = clt.score(X_test, y_test)
    print("准确率为：", curr_score)
    score = score + curr_score

avg_score = score / 5
print("平均准确率为：", avg_score)

# 法2====主要用于比较多个模型，从而选择最佳模型
kf = KFold(n_splits=5, shuffle=True, random_state=0)
model1 = DecisionTreeClassifier(max_depth=5, random_state=0)
scores_model1 = cross_val_score(model1, X, y, cv=kf)
print("准确率为：", scores_model1)
print("平均准确率为：", scores_model1.mean())



# 法1====主要用于找出单个模型最佳的数据集划分
kf = KFold(n_splits = 5, shuffle=True, random_state=0)
score=0
best_score = float('inf')  # 初始化为正无穷大
best_train_index = None
best_test_index = None
for train_index, test_index in kf.split(y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clt = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
    curr_score = clt.score(X_test, y_test)
    print("准确率为：", curr_score)
    score = score + curr_score

    # 如果当前数据集划分的性能更好，则更新最佳数据集划分
    if curr_score < best_score:
        best_score = curr_score
        best_train_index = train_index
        best_test_index = test_index

avg_score = score / 5
print("平均准确率为：", avg_score)
print("Best train indices:", best_train_index)
print("Best test indices:", best_test_index)


# 法2====主要用于比较多个模型，选择一个最佳模型
kf = KFold(n_splits=5, shuffle=True, random_state=0)
model1 = DecisionTreeClassifier(max_depth=5, random_state=0)
model2 = LogisticRegression()
scores_model1 = cross_val_score(model1, X, y, cv=kf)
scores_model2 = cross_val_score(model2, X, y, cv=kf)
print("model1准确率为：", scores_model1)
print("model1平均准确率为：", scores_model1.mean())
print("model2准确率为：", scores_model2)
print("model2平均准确率为：", scores_model2.mean())
