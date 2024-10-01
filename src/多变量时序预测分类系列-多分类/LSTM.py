#导入必要的库
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
import shap
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from keras.utils import np_utils
import seaborn as sns

#读取数据
df = pd.read_excel('data.xlsx')
df = pd.DataFrame(df)
df = df.drop(columns=['时间变化(s)'])
df.insert(0, '标签', df.pop('标签'))
fea_num = len(df.columns)
print(df)

# 数据划分
test_split = round(len(df)*0.20)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]

# 实例化 LabelEncoder
encoder = LabelEncoder()
df_for_training['标签'] = encoder.fit_transform(df_for_training['标签'])
df_for_testing['标签'] = encoder.transform(df_for_testing['标签'])
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

# 标签进行进一步编码
trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

#建立LSTM模型 训练
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, fea_num), return_sequences=False))
model.add(Dropout(0.01))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs = 20, batch_size = 200,validation_data=(testX, testY)) #训练模型1000次


# 预测
prediction_train=model.predict(trainX)
prediction_train = np.argmax(prediction_train,axis=1)
really_train = np.argmax(trainY,axis=1)
print(classification_report(really_train,prediction_train))
print(confusion_matrix(really_train,prediction_train))

# 1.计算混淆矩阵
conf_matrix = pd.DataFrame(confusion_matrix(really_train,prediction_train), index=['1', '2', '3', '4'], columns=['1', '2', '3', '4'])  # 数据有5个类别
# 画出混淆矩阵
print(conf_matrix)
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt=".0f", annot_kws={"size": 14}, cmap="Blues")
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()



prediction_test=model.predict(testX)
prediction_test = np.argmax(prediction_test,axis=1)
really_test = np.argmax(testY,axis=1)
print(classification_report(really_test,prediction_test))
print(confusion_matrix(really_test,prediction_test))
# 1.计算混淆矩阵
conf_matrix = pd.DataFrame(confusion_matrix(really_test,prediction_test), index=['1', '2', '3', '4'], columns=['1', '2', '3', '4'])  # 数据有5个类别
# 画出混淆矩阵
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt=".0f", annot_kws={"size": 14}, cmap="Blues")
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

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


'''
###============shap分析================================
print(trainX[:100])
trainX_shap_smaple = trainX[:100].reshape(100, window_size*fea_num)
print(trainX_shap_smaple)


explainer = shap.GradientExplainer(model, trainX[:100])

# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(trainX[:100])
# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(trainX[:100])
print(shap_obj)
print(shap_obj.shape)
print(shap_obj[:,:,:,0].shape)


shap.plots.bar(shap_obj[:,:,:,0], show=True)        # 全局条形图
shap.plots.beeswarm(shap_obj[:,:,:,0], show=True)   # 全局蜂群图
'''