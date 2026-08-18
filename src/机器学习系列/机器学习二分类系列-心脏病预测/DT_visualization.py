import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pybaobabdt
import seaborn as sns
import matplotlib.pyplot as plt

# 1、准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df.head())

# 2、 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

# 3、划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 4、模型的构建与训练
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# 5、决策树可视化
sns.set(font_scale=1.2)
plt.rc('font', family=['Simsun'], size=12)
ax = pybaobabdt.drawTree(model, size=10, dpi=300, features=features) #可视化主函数pybaobabdt.drawTree
plt.show()

# # 9、模型推理与评价
# y_pred = model.predict(X_test)
# y_scores = model.predict_proba(X_test)
# acc = accuracy_score(y_test, y_pred) # 准确率acc
# cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
# cr = classification_report(y_test, y_pred) # 分类报告
# fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
