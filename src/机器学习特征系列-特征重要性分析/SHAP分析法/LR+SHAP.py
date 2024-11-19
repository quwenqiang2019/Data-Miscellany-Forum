import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df)
# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)

mm1 = MinMaxScaler()  # 特征进行归一化
X_train_m = mm1.fit_transform(X_train)
mm2 = MinMaxScaler()  # 标签进行归一化
y_train_m = mm2.fit_transform(y_train)

# 模型的构建与训练
model = LogisticRegression()
model.fit(X_train_m, y_train_m)

X_test_m = mm1.transform(X_test)
print(X_test_m.shape)
# 创建Explainer
explainer = shap.KernelExplainer(model.predict, X_test_m)


# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(X_test_m)
print(shap_values)     # shap_values = shap_obj.values
print(shap_values.shape)   # (60, 13)

# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(X_test_m)
print(shap_obj.values)
print(shap_obj.shape)  # (60, 13)

# 特征分析
# shap.plots.bar(shap_obj, show=True)        # 全局条形图
# shap.plots.beeswarm(shap_obj, show=True)   # 全局蜂群图

shap.summary_plot(shap_obj, X_test_m,feature_names=features)
shap.summary_plot(shap_obj, X_test_m, plot_type="bar",feature_names=features)



