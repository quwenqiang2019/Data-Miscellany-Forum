from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 加载数据集
boston=load_boston()
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df['target']=boston.target
#查看数据项
features=df[boston.feature_names]
target=df['target']

#数据集划分
split_num=int(len(features)*0.8)
X_train=features[:split_num]
Y_train=target[:split_num]
X_test=features[split_num:]
Y_test=target[split_num:]

# 决策树建模预测
regressor = DecisionTreeRegressor(random_state=0).fit(X_train,Y_train)
y_pred=regressor.predict(X_test)

# 可视化部分
sns.set(font_scale=1.2)
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',size=14)

plt.plot(list(range(0,len(X_test))),Y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('Boston房价决策回归树预测值与真实值的对比')
plt.show()