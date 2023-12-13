from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
data, target = iris.data, iris.target
print(data)
print(target)

kf = KFold(n_splits = 5, shuffle=True, random_state=0)
curr_score=0
for train_index, test_index in kf.split(data):
    clt = DecisionTreeClassifier(max_depth=5, random_state=0).fit(data[train_index], target[train_index])
    curr_score = curr_score + clt.score(data[test_index], target[test_index])
    print("准确率为：", clt.score(data[test_index], target[test_index]))

avg_score = curr_score / 5
print("平均准确率为：", avg_score)
