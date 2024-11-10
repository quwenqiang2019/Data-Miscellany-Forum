import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
from xgboost.sklearn import XGBClassifier

# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df)
# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 模型的构建与训练
model = XGBClassifier()
model.fit(X_train, y_train)

print(X_test.shape)
# 创建Explainer
explainer = shap.TreeExplainer(model, X_test)


# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(X_test)
print(shap_values)     # shap_values = shap_obj.values
print(shap_values.shape)

# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(X_test)
print(shap_obj.values)    # shap_values = shap_obj.values
print(shap_obj.shape) 

# 特征分析
shap.plots.bar(shap_obj, show=True)        # 全局条形图
shap.plots.beeswarm(shap_obj, show=True)   # 全局蜂群图
