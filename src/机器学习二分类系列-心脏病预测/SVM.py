import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
## 数据基本信息
print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.dtypes)
cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名

# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

# 划分训练集和测试集
# df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)

# 归一化
mm1 = MinMaxScaler()   # 特征进行归一化
X_train_m = mm1.fit_transform(X_train)
mm2 = MinMaxScaler()     # 标签进行归一化
y_train_m = mm2.fit_transform(y_train)


# 模型的构建与训练
model = SVC(probability=True)
model.fit(X_train, y_train)

# 模型推理与评价
# 对测试集特征进行相同规则mm1的归一化处理，然后输入到模型进行预测
X_test_m = mm1.transform(X_test) #注意fit_transform() 和 transform()的区别
y_pred_m = model.predict(X_test_m) #利用输入特征input1和input2测试模型
y_scores = model.predict_proba(X_test_m)
y_pred = mm2.inverse_transform(np.reshape(y_pred_m, (-1, 1)))

y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred) # 准确率acc
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
cr = classification_report(y_test, y_pred) # 分类报告
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
