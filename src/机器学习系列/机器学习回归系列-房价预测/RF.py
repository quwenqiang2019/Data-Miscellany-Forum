import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


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
model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 可视化部分
sns.set(font_scale=1.2)
plt.rc('font', family=['SimSun'], size=12)
# 训练集预测值与真实值的对比
plt.plot(list(range(0,len(X_train))),y_train,marker='o')
plt.plot(list(range(0,len(X_train))),y_train_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.xlabel('序列')
plt.ylabel('房价')
plt.title('训练集预测值与真实值的对比')
plt.show()
# 验证集预测值与真实值的对比
plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_test_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.xlabel('序列')
plt.ylabel('房价')
plt.title('验证集预测值与真实值的对比')
plt.show()

# 评价指标
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