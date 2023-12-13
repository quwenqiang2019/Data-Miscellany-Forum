from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn import preprocessing
# 加载数据集
boston=load_boston()
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df['target']=boston.target
#查看数据项
features=df[boston.feature_names]
target=df['target']
print(df)
print(features)
print(target)
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
print(features)


#数据集划分
split_num=int(len(features)*0.8)
X_train=features[:split_num]
Y_train=target[:split_num]
X_test=features[split_num:]
Y_test=target[split_num:]

# 多项式回归建模预测
boston_poly=PolynomialFeatures(2)
boston_poly.fit(X_train)
X_train2=boston_poly.transform(X_train)
print('原始数据集X的形状为：',X_train.shape)
print('X转换为X2后的形状为：',X_train2.shape)
X_test2=boston_poly.transform(X_test)


lin_reg=LinearRegression().fit(X_train2,Y_train)
y_reg_pred=lin_reg.predict(X_test2)

# 可视化部分
sns.set(font_scale=1.2)
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',size=14)

plt.plot(list(range(0,len(X_test))),Y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_reg_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('Boston房价多项式回归预测值与真实值的对比')
plt.show()