import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import shap

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
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(X_test.shape)
# 创建Explainer
explainer = shap.TreeExplainer(model, X_test)


# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(X_test)
print(shap_values)     # shap_values = shap_obj.values
print(shap_values.shape)   # (60, 13, 2)
# print(shap_values[0].shape)
# print(shap_values[0][0].shape)
# print(shap_values[0][0][0].shape)
print(shap_values[:,:,0].shape)  # (60, 13)

# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(X_test)
print(shap_obj.values)
print(shap_obj.shape)  # (60, 13, 2)
print(shap_obj[:,:,0].shape) # (60, 13)


# 特征分析
shap.plots.bar(shap_obj[:,:,0], show=True)        # 全局条形图
shap.plots.beeswarm(shap_obj[:,:,0], show=True)   # 全局蜂群图
# shap.plots.beeswarm(shap_obj[:,:,1], show=True)   # 全局蜂群图
# shap.plots.force(explainer.expected_value[0], shap_obj.values[0,:][:, 1], np.array(X_test.iloc[0,:]), matplotlib=True, show=True, feature_names=features)   # 单个样本力图
shap.plots.waterfall(shap_obj[0,:,1])    # 单个样本瀑布图
print(shap_obj[0,:,1])

