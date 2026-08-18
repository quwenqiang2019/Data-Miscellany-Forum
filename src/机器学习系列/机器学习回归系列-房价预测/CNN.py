import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense

# 导入数据
filename = 'housing.csv'
# 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename, names=names, delim_whitespace=True)
print(dataset)
df = pd.DataFrame(dataset)

#查看数据项
features = names[:-1]
target = ['MEDV']
#  划分数据集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

model = Sequential()
model.add(Conv1D(16, 3, input_shape=(13, 1), activation='relu'))
model.add(Conv1D(32, 3, activation='relu'))
# model.add(MaxPooling1D(3))
model.add(Conv1D(32, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8)

scores = model.evaluate(X_test, y_test, verbose=0)
print('accuracy:%.2f%%' % (scores[1] * 100))
predicted = model.predict(X_test)
result = abs(np.mean(predicted - y_test))
print(result)

