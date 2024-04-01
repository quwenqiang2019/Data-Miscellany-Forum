import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 准备数据
data = pd.read_csv(r'data.csv')
df = pd.DataFrame(data)
print(df.describe())


# 提取目标变量和特征变量
target = 'Type'
features = df.columns.drop(target)
print(data["Type"].value_counts()) # 顺便查看一下样本是否平衡


# 特征重要性分析与筛选


# 划分训练集和测试集
df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 模型的构建与训练
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# 模型推理与评价
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
print(cm)
cr = classification_report(y_test, y_pred) # 分类报告
print(cr)
