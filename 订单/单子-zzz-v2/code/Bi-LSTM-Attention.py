#导入必要的库
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from keras.utils import np_utils
import seaborn as sns
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention, Multiply, Flatten, Bidirectional
from keras.layers import *
from keras.models import *

#读取数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
df1=pd.read_excel(os.path.join(base_dir, 'data', '指标特征.xlsx'))
df1 = pd.DataFrame(df1)
# df1 = df1.drop(df1.columns[0], axis=1)
df = df1.dropna(axis=0, how='any')
df.insert(0, '标签', df.pop('标签'))
print(df)
fea_num = len(df.columns)
fea_name = df.columns
print(fea_name)

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

window_size = 1
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


#建立Bi-LSTM-Attention模型 训练
inputs=Input(shape=(window_size, fea_num))
model=Bidirectional(LSTM(50, activation='tanh'))(inputs)
attention=Dense(100, activation='sigmoid', name='attention_vec')(model)#求解Attention权重
model=Multiply()([model, attention])#attention与LSTM对应数值相乘
outputs = Dense(2, activation='softmax')(model)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs = 100, batch_size = 200,validation_data=(testX, testY)) #训练模型1000次

# 预测
prediction_train=model.predict(trainX)
prediction_train = np.argmax(prediction_train,axis=1)
really_train = np.argmax(trainY,axis=1)
print(classification_report(really_train,prediction_train))
print(confusion_matrix(really_train,prediction_train))

# 1.计算混淆矩阵
conf_matrix = pd.DataFrame(confusion_matrix(really_train,prediction_train), index=['1', '2'], columns=['1', '2'])  # 数据有5个类别
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
plt.savefig(os.path.join(base_dir, 'result', 'bi_lstm_attention_train_conf_matrix.jpg'), bbox_inches='tight', dpi = 600)
plt.show()



prediction_test=model.predict(testX)
prediction_test = np.argmax(prediction_test,axis=1)
really_test = np.argmax(testY,axis=1)
print(classification_report(really_test,prediction_test))
print(confusion_matrix(really_test,prediction_test))
# 1.计算混淆矩阵
conf_matrix = pd.DataFrame(confusion_matrix(really_test,prediction_test), index=['1', '2'], columns=['1', '2'])  # 数据有5个类别
# 画出混淆矩阵
sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt=".0f", annot_kws={"size": 14}, cmap="Blues")
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(base_dir, 'result', 'bi_lstm_attention_test_conf_matrix.jpg'), bbox_inches='tight', dpi = 600)
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
plt.savefig(os.path.join(base_dir, 'result', 'bi_lstm_attention_loss.jpg'), bbox_inches='tight', dpi = 600)
plt.show()

plt.figure()
plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.savefig(os.path.join(base_dir, 'result', 'bi_lstm_attention_accuracy.jpg'), bbox_inches='tight', dpi = 600)
plt.show()



###============shap分析================================
train_sample = trainX[:100]
print(train_sample.shape)  # (100, 2, 9)
trainX_shap_smaple = train_sample.reshape(100, window_size*fea_num)
print(trainX_shap_smaple.shape)  # (100, 18)


explainer = shap.GradientExplainer(model, train_sample)

# 以numpy数组的形式输出SHAP值
shap_values = explainer.shap_values(train_sample)
print(shap_values.shape) # (100, 2, 9, 4)
print(shap_values.reshape(100, window_size*fea_num, 2).shape)  # (100, 18, 4)

# # 以SHAP的Explanation对象形式输出SHAP值
shap_obj = explainer(train_sample)
print(shap_obj)
print(shap_obj.shape) # (100, 2, 9, 4)
print(shap_obj[:,:,:,0].shape) # (100, 2, 9)


##### shap.summary_plot(shap_values.reshape(100, 2*9, 4), trainX_shap_smaple)
shap.summary_plot(shap_obj[:,:,:,0].values.reshape(100, window_size*fea_num), trainX_shap_smaple,feature_names=fea_name)
# plt.savefig(os.path.join(base_dir, 'result', 'summary_1.png'), bbox_inches='tight', dpi=600)
##### shap.summary_plot(shap_values.reshape(100, 2*9, 4), trainX_shap_smaple, plot_type="bar")
shap.summary_plot(shap_obj[:,:,:,0].values.reshape(100, window_size*fea_num), trainX_shap_smaple, plot_type="bar",feature_names=fea_name)
# plt.savefig(os.path.join(base_dir, 'result', 'bi_lstm_attention_summary_2.png'), bbox_inches='tight', dpi=600)


