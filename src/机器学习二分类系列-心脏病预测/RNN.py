import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from keras import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN


# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
## 数据基本信息
print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.dtypes)
cat_cols = [col for col in df.columns if df[col].dtype == "object"] # 类别型变量名
num_cols = [col for col in df.columns if df[col].dtype != "object"] # 数值型变量名

# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

# 划分训练集和测试集
df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)


# 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# 指定了变换后的数组的形状，其分别为（样本数量，特征数量，1），其中1表示数据的通道数。
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# 同样指定了变换后的测试集数组的形状，其也为（样本数量，特征数量，1）。
print(X_train, y_train)



model = Sequential()
model.add(SimpleRNN(200, input_shape=(13, 1), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


opt= tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics="accuracy")


epochs=100
history=model.fit(X_train,y_train,
                  epochs=epochs,
                  batch_size=128,
                  validation_data=(X_test,y_test),
                  verbose=1)



import matplotlib.pyplot as plt

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(epochs)

plt.figure(figsize=(14,4))
plt.subplot(1,2,1)

plt.plot(epochs_range, acc,label='Training Accuracy')
plt.plot(epochs_range, val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss,label='Training Loss')
plt.plot(epochs_range, val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



