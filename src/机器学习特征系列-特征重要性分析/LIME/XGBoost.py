import pandas as pd
import math
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from lime.lime_tabular import LimeTabularExplainer

# 导入数据
filename = 'data.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
print(dataset)
df = pd.DataFrame(dataset)

#  划分数据集
features = names[:-1]
target = ['MEDV']
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 建模预测
model = XGBRegressor(n_estimators=100, max_depth=10)
help(model)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print(r2_score(y_test, y_test_pred))


feature_names = names
#建立解释器
explainer = LimeTabularExplainer(X_train,feature_names=feature_names,mode='regression')
# 解释第81个样本的规则,选择10个特征
exp = explainer.explain_instance(X_test[81], model.predict,num_features=5)
# 画图
fig = exp.as_pyplot_figure()