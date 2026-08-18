#导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, GRU
from keras.models import *

# Open为开盘价，
# High为当天最高价，
# Low为最低价，
# Close为当天收盘价，
# Adj Close为调整后收盘价（比如派发分红，或拆股后进行调整），
# Volume指当天总共的股票交易数。

#读取数据
df1 = pd.read_table("train-small.txt",sep=',',header=0)
df1 = df1.iloc[:10000,:]
# 将Time (UTC)列设置为索引
df1.set_index('Time (UTC)', inplace=True)
print(df1)

#进行数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
df0 = min_max_scaler.fit_transform(df1)
df1 = pd.DataFrame(df0, columns=df1.columns)

#计算得出标签
record=(df1['Close'][1:].values-df1['Close'][0:-1].values)>0
classification=[0]
for i in record:
    if(i==True):
        classification.append(1)
    else:
        classification.append(0)

df1['label']=classification
df1.insert(0, 'label', df1.pop('label'))
fea_num = len(df1.columns)
print(df1)

# 数据划分
df = df1
test_split = round(len(df)*0.20)
print(test_split)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
df_for_training=df_for_training.values
df_for_testing=df_for_testing.values

# 数据构造
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

# 将数据集转换为模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

#建立模型 训练
model = Sequential([
    GRU(80, return_sequences=True),
    Dropout(0.2),
    GRU(100),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs = 20, batch_size = 200,validation_data=(testX, testY))

#画出迭代曲线
hist = pd.DataFrame(history.history)
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
plt.plot(hist['loss'], label='Train Loss')
plt.plot(hist['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()


#在训练集上的拟合结果
y_train_predict=model.predict(trainX)
y_train_predict=y_train_predict[:,0]
print(y_train_predict)
print(y_train_predict>0.5)
y_train_predict=[int(i) for i in y_train_predict>0.5]
y_train_predict=np.array(y_train_predict)

print("精确度等指标：")
print(metrics.classification_report(trainY,y_train_predict))
print("混淆矩阵：")
print(metrics.confusion_matrix(trainY,y_train_predict))


#在测试集上的拟合结果
y_test_predict=model.predict(testX)
y_test_predict=y_test_predict[:,0]
print(y_test_predict)
print(y_test_predict>0.5)
y_test_predict=[int(i) for i in y_test_predict>0.5]
y_test_predict=np.array(y_test_predict)

print("精确度等指标：")
print(metrics.classification_report(testY,y_test_predict))
print("混淆矩阵：")
print(metrics.confusion_matrix(testY,y_test_predict))
conf_matrix = pd.DataFrame(metrics.confusion_matrix(testY,y_test_predict), index=['跌', '涨'], columns=['跌', '涨'])
# 画出混淆矩阵
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt=".0f",  annot_kws={"size": 12}, cmap="Blues")
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()