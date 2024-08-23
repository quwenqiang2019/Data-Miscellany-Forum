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
df = pd.DataFrame(dataset)
print(df)

#  划分数据集
features = names[:-1]
target = ['MEDV']
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# 建模预测
model = XGBRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print(r2_score(y_test, y_test_pred))

# 建立解释器，解释第1个样本的规则,选择13个特征
explainer = LimeTabularExplainer(X_train.values,feature_names=list(X_train.columns), mode='regression')
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict, num_features=13)
# 画图
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
# 需要在jupyter notebook中运行才能显示
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('oi.html')


