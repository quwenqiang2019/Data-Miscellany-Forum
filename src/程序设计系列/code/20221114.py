from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# 辅助函数
def cross_val(model,X,Y):
    pred = cross_val_score(model, X, Y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

# 加载数据集
boston=load_boston()
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df['target']=boston.target
#查看数据项
features=df[boston.feature_names]
target=df['target']
#数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
#数据集划分
split_num=int(len(features)*0.8)
X_train=features[:split_num]
Y_train=target[:split_num]
X_test=features[split_num:]
Y_test=target[split_num:]
#多层感知机（神经网络）建模
model = Sequential()
model.add(Dense(X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer=Adam(0.00001), loss='mse')
# 训练以及验证
r = model.fit(X_train, Y_train,validation_data=(X_test, Y_test),batch_size=1,epochs=100)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(Y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(Y_train, train_pred)
# 损失函数曲线
df1=pd.DataFrame(r.history)
plt.plot([x for x in range(100)],df1['loss'])
plt.plot([x for x in range(100)],df1['val_loss'])
plt.show()
# 可视化部分
sns.set(font_scale=1.2)
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',size=14)
plt.plot(list(range(0,len(X_test))),Y_test,marker='o')
plt.plot(list(range(0,len(X_test))),test_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('Boston房价多层感知机（神经网络）预测值与真实值的对比')
plt.show()