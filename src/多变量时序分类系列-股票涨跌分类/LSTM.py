#导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM


#读取数据
df1=pd.read_table("train-small.txt",sep=',',header=0)
df1=df1.iloc[:10000,1:]
print(df1.tail())

#进行数据归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df1 = pd.DataFrame(df0, columns=df1.columns)


#调整列顺序
cols=list(df1)
cols.insert(0,cols.pop(cols.index('Volume ')))
df1=df1[cols]

#计算得出标签
record=(df1['Close'][1:].values-df1['Close'][0:-1].values)>0
classification=[0]
for i in record:
    if(i==True):
        classification.append(1)
    else:
        classification.append(0)
print(classification)
df1['label']=classification
df1.insert(0, 'label', df1.pop('label'))
print(df1)
fea_num = len(df1.columns)


# 数据划分
df = df1
test_split = round(len(df)*0.20)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
df_for_training=df_for_training.values
df_for_testing=df_for_testing.values

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i,0])

    return np.array(dataX),np.array(dataY)

window_size = 2
trainX,trainY=createXY(df_for_training,window_size)
testX,testY=createXY(df_for_testing,window_size)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

#建立LSTM模型 训练
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, fea_num), return_sequences=False))
model.add(Dropout(0.01))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs = 100, batch_size = 200,validation_data=(testX, testY)) #训练模型1000次


#画出迭代曲线
pd.DataFrame(model.history.history).plot()
plt.show()


#在训练集上的拟合结果
y_train_predict=model.predict(trainX)
y_train_predict=y_train_predict[:,0]
print(y_train_predict>0.5)
y_train_predict=[int(i) for i in y_train_predict>0.5]
y_train_predict=np.array(y_train_predict)
from sklearn import metrics
print("精确度等指标：")
print(metrics.classification_report(trainY,y_train_predict))
print("混淆矩阵：")
print(metrics.confusion_matrix(trainY,y_train_predict))