import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve

# 准备数据
data = pd.read_csv(r'data.csv')
df = pd.DataFrame(data)
print(df.head())
print(df.describe())


# 提取目标变量和特征变量
target = 'Type'
features = df.columns.drop(target)
print(data["Type"].value_counts()) # 顺便查看一下样本是否平衡


# 划分训练集和测试集
# df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# 对于标签采用LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# 归一化
mm1 = MinMaxScaler()   # 特征进行归一化
X_train_m = mm1.fit_transform(X_train)

# 模型的构建与训练
model = AdaBoostClassifier()
model.fit(X_train_m, y_train)

# 模型推理与评价
X_test_m = mm1.transform(X_test)
y_test = le.transform(y_test)
y_pred = model.predict(X_test_m)
y_scores = model.predict_proba(X_test_m)
print(y_pred)
acc = accuracy_score(y_test, y_pred) # 准确率acc
print(f"acc: \n{acc}")
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
print(f"cm: \n{cm}")
cr = classification_report(y_test, y_pred) # 分类报告
print(f"cr:  \n{cr}")
# 绘制混淆矩阵热力图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
sns.heatmap(cm, annot=True, annot_kws={'size':15},
            fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion matrix heat map', fontsize=15)
plt.show()
cr = classification_report(y_test, y_pred) # 分类报告
print(f"cr:  \n{cr}")


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


# 将y标签转换成one-hot形式
ytest_one_rf = label_binarize(y_test, classes=[0, 1, 2])

# 宏平均法计算AUC
rf_AUC = {}
rf_FPR = {}
rf_TPR = {}

for i in range(ytest_one_rf.shape[1]):
    rf_FPR[i], rf_TPR[i], thresholds = roc_curve(ytest_one_rf[:, i], y_scores[:, i])
    rf_AUC[i] = auc(rf_FPR[i], rf_TPR[i])
print(rf_AUC)

# 合并所有的FPR并排序去重
rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(ytest_one_rf.shape[1])]))

# 计算宏平均TPR
rf_TPR_all = np.zeros_like(rf_FPR_final)
for i in range(ytest_one_rf.shape[1]):
    rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]

# 计算最终的宏平均AUC
rf_AUC_final = auc(rf_FPR_final, rf_TPR_final)
AUC_final_rf = rf_AUC_final  # 最终AUC

print(f"Macro Average AUC with Random Forest: {AUC_final_rf}")

plt.figure()
# 使用不同的颜色和线型
plt.plot(rf_FPR[0], rf_TPR[0], color='#1f77b4', linestyle='-', label='Class 1 ROC  AUC={:.4f}'.format(rf_AUC[0]), lw=2)
plt.plot(rf_FPR[1], rf_TPR[1], color='#ff7f0e', linestyle='-', label='Class 2 ROC  AUC={:.4f}'.format(rf_AUC[1]), lw=2)
plt.plot(rf_FPR[2], rf_TPR[2], color='#2ca02c', linestyle='-', label='Class 3 ROC  AUC={:.4f}'.format(rf_AUC[2]), lw=2)

# 宏平均ROC曲线
plt.plot(rf_FPR_final, rf_TPR_final, color='#000000', linestyle='-', label='Macro Average ROC  AUC={:.4f}'.format(rf_AUC_final), lw=3)
# 45度参考线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='45 Degree Reference Line')
plt.xlabel('False Positive Rate (FPR)', fontsize=15)
plt.ylabel('True Positive Rate (TPR)', fontsize=15)
plt.title('Random Forest Classification ROC Curves and AUC', fontsize=18)
plt.grid(linestyle='--', alpha=0.7)
plt.legend(loc='lower right', framealpha=0.9, fontsize=12)
plt.show()