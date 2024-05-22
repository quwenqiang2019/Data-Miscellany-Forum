import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import shap

# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)

# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 模型的构建与训练
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 创建Explainer
explainer = shap.TreeExplainer(model, X_test)
# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(X_test)
print(shap_values)     # shap_values = shap_obj.values
# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(X_test)
print(shap_obj.values)


# 特征分析
# shap.plots.bar(shap_obj[:,:,0], show=True)        # 全局条形图
# shap.plots.beeswarm(shap_obj[:,:,0], show=True)   # 全局蜂群图
# shap.plots.beeswarm(shap_obj[:,:,1], show=True)   # 全局蜂群图
shap.plots.force(explainer.expected_value[0], shap_obj.values[0,:][:, 1], np.array(X_test.iloc[0,:]), matplotlib=True, show=True, feature_names=features)   # 单个样本力图
# shap.plots.waterfall(shap_obj[0,:,1])    # 单个样本瀑布图




# # 模型推理与评价
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
#
