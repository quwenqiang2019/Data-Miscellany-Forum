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
from sklearn.metrics import precision_score, recall_score, f1_score

# 准备数据
data = pd.read_csv(r'data.csv')
df = pd.DataFrame(data)
print(df.describe())


# 提取目标变量和特征变量
target = 'Type'
features = df.columns.drop(target)
print(data["Type"].value_counts()) # 顺便查看一下样本是否平衡


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
print(acc)
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
print(cm)
cr = classification_report(y_test, y_pred) # 分类报告
print(cr)


print("----------------------------- precision（精确率）-----------------------------")
precision_score_average_None = precision_score(y_test, y_pred, average=None)
precision_score_average_micro = precision_score(y_test, y_pred, average='micro')
precision_score_average_macro = precision_score(y_test, y_pred, average='macro')
precision_score_average_weighted = precision_score(y_test, y_pred, average='weighted')
print('precision_score_average_None = ', precision_score_average_None)
print('precision_score_average_micro = ', precision_score_average_micro)
print('precision_score_average_macro = ', precision_score_average_macro)
print('precision_score_average_weighted = ', precision_score_average_weighted)

print("\n\n----------------------------- recall（召回率）-----------------------------")
recall_score_average_None = recall_score(y_test, y_pred, average=None)
recall_score_average_micro = recall_score(y_test, y_pred, average='micro')
recall_score_average_macro = recall_score(y_test, y_pred, average='macro')
recall_score_average_weighted = recall_score(y_test, y_pred, average='weighted')
print('recall_score_average_None = ', recall_score_average_None)
print('recall_score_average_micro = ', recall_score_average_micro)
print('recall_score_average_macro = ', recall_score_average_macro)
print('recall_score_average_weighted = ', recall_score_average_weighted)

print("\n\n----------------------------- F1-value-----------------------------")
f1_score_average_None = f1_score(y_test, y_pred, average=None)
f1_score_average_micro = f1_score(y_test, y_pred, average='micro')
f1_score_average_macro = f1_score(y_test, y_pred, average='macro')
f1_score_average_weighted = f1_score(y_test, y_pred, average='weighted')
print('f1_score_average_None = ', f1_score_average_None)
print('f1_score_average_micro = ', f1_score_average_micro)
print('f1_score_average_macro = ', f1_score_average_macro)
print('f1_score_average_weighted = ', f1_score_average_weighted)