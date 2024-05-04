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
data = pd.read_csv(r'dataset.csv')
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
# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(X_test)


# 特征分析
# shap.summary_plot(shap_values, X_test, show=True)
shap.plots.bar(shap_obj[:,:,0], show=True)
shap.plots.beeswarm(shap_obj[:,:,0], show=True)
shap.plots.beeswarm(shap_obj[:,:,1], show=True)

