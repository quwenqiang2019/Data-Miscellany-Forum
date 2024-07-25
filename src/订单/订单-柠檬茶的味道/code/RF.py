import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

# 读取数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'fitness.csv'))
df = pd.DataFrame(data)

# 数据划分
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(train_df)
print(test_df)


# 训练集预处理（编码和归一化）
target = 'RR'
processing_col = ['Polymer type','Particle Shape']
no_processing_col = ['Concentration', 'Size','CI', 'HI', 'FI', 'Day']
ohe = OneHotEncoder(sparse=False)  # 对于类别特征采用OneHotEncoder
train_df_X = pd.DataFrame(ohe.fit_transform(train_df[processing_col].values), columns=ohe.get_feature_names())
train_df = pd.concat([train_df_X, train_df[no_processing_col], train_df[target]], axis=1)
print(train_df)
X_train = train_df.iloc[:,:-1]
y_train = train_df['RR']

# 模型的构建与训练
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 测试集做和训练集相同的预处理
test_df_X = pd.DataFrame(ohe.transform(test_df[processing_col].values), columns=ohe.get_feature_names())
test_df = pd.concat([test_df_X, test_df[no_processing_col], test_df[target]], axis=1)
print(test_df)
X_test = test_df.iloc[:,:-1]
y_test = test_df['RR']


# # 模型推理与评价
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 可视化部分
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
# 训练集预测值与真实值的对比
plt.plot(list(range(0,len(X_train))),y_train,marker='o')
plt.plot(list(range(0,len(X_train))),y_train_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.xlabel('序列')
plt.ylabel(target)
plt.title('训练集预测值与真实值的对比')
plt.savefig(os.path.join(base_dir, 'result', 'train_result.png'), bbox_inches='tight', dpi=600)
plt.show()
# 验证集预测值与真实值的对比
plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_test_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.xlabel('序列')
plt.ylabel(target)
plt.title('验证集预测值与真实值的对比')
plt.savefig(os.path.join(base_dir, 'result', 'test_result.png'), bbox_inches='tight', dpi=600)
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
