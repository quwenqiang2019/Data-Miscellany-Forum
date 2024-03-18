# 导入类库
import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
import pandas as pd
import math
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_percentage_error
# 导入数据
filename = 'housing.csv'
# 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
print(dataset)
df = pd.DataFrame(dataset)

#查看数据项
features = names[:-1]
target = ['MEDV']

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


# 决策树建模预测
model = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 可视化部分
sns.set(font_scale=1.2)
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',size=14)

plt.plot(list(range(0,len(X_train))),y_train,marker='o')
plt.plot(list(range(0,len(X_train))),y_train_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('Boston房价决策回归树预测值与真实值的对比')
plt.show()



sns.set(font_scale=1.2)
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',size=14)

plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_test_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('Boston房价决策回归树预测值与真实值的对比')
plt.show()

# 计算误差
trainScore1 = math.sqrt(mean_squared_error(y_train, y_train_pred))
print('Train Score: %.2f RMSE' % (trainScore1))
testScore1 = math.sqrt(mean_squared_error(y_test, y_test_pred))
print('Test Score: %.2f RMSE' % (testScore1))

trainScore2 = mean_absolute_error(y_train, y_train_pred)
print('Train Score: %.2f MAE' % (trainScore2))
testScore2 = mean_absolute_error(y_test, y_test_pred)
print('Test Score: %.2f MAE' % (testScore2))

trainScore3 = r2_score(y_train, y_train_pred)
print('Train Score: %.2f R2' % (trainScore3))
testScore3 = r2_score(y_test, y_test_pred)
print('Test Score: %.2f R2' % (testScore3))

trainScore4 = mean_absolute_percentage_error(y_train, y_train_pred)
print('Train Score: %.2f MAPE' % (trainScore4))
testScore4 = mean_absolute_percentage_error(y_test, y_test_pred)
print('Test Score: %.2f MAPE' % (testScore4))