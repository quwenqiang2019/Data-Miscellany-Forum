import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
df = pd.read_csv(os.path.join(base_dir, 'data', "2_300股指加权(日线).csv"), encoding = 'gb2312')

df = pd.DataFrame(df)
df = df.iloc[:, 0:8]
df['DateTime'] = pd.to_datetime(df['日期'])
df.drop(['日期'], axis=1, inplace=True)
df.set_index('DateTime', inplace = True)
df.insert(0, '开盘', df.pop('开盘'))
print(df.shape)
print(df.head())
fea_num = len(df.columns)

test_split=round(len(df)*0.20)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]


# 绘制训练集和测试集的折线图
sns.set_style('darkgrid')
font1 = {'family': ['SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6))
plt.plot(df_for_training['开盘'], label='Training Data')
plt.plot(df_for_testing['开盘'], label='Testing Data')
plt.xlabel('时间')
plt.xticks(rotation=45)
plt.ylabel('开盘')
plt.title('开盘')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'Train_and_Test.jpg'), bbox_inches='tight', dpi = 600)
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)


def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

window_size = 1
trainX,trainY=createXY(df_for_training_scaled,window_size)
testX,testY=createXY(df_for_testing_scaled,window_size)

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)


def build_model():
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(window_size,fea_num)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return grid_model

grid_model = KerasRegressor(model=build_model,verbose=1)
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
grid_search = grid_search.fit(trainX,trainY)
print(grid_search.best_params_)
my_model=grid_search.best_estimator_


prediction_test=my_model.predict(testX)
prediction_train=my_model.predict(trainX)

prediction_train_copies_array = np.repeat(prediction_train,fea_num, axis=-1)
pred_train=scaler.inverse_transform(np.reshape(prediction_train_copies_array,(len(prediction_train),fea_num)))[:,0]
original_train_copies_array = np.repeat(trainY, fea_num, axis=-1)
original_train=scaler.inverse_transform(np.reshape(original_train_copies_array,(len(trainY),fea_num)))[:,0]
print("train Pred Values-- ", pred_train)
print("\ntrain Original Values-- ", original_train)
plt.plot(df_for_training.index[window_size:,], original_train, color = 'red', label = '真实值')
plt.plot(df_for_training.index[window_size:,], pred_train, color = 'blue', label = '预测值')
plt.title('开盘预测')
plt.xlabel('时间')
plt.xticks(rotation=45)
plt.ylabel('开盘')
plt.legend()
plt.show()


prediction_test_copies_array = np.repeat(prediction_test,fea_num, axis=-1)
pred_test=scaler.inverse_transform(np.reshape(prediction_test_copies_array,(len(prediction_test),fea_num)))[:,0]
original_test_copies_array = np.repeat(testY, fea_num, axis=-1)
original_test=scaler.inverse_transform(np.reshape(original_test_copies_array,(len(testY),fea_num)))[:,0]
print("test Pred Values-- ", pred_test)
print("\ntest Original Values-- ", original_test)
plt.plot(df_for_testing.index[window_size:,], original_test, color = 'red', label = '真实值')
plt.plot(df_for_testing.index[window_size:,], pred_test, color = 'blue', label = '预测值')
plt.title('开盘预测')
plt.xlabel('时间')
plt.xticks(rotation=45)
plt.ylabel('开盘')
plt.legend()
plt.show()

# 计算误差
testScore1 = math.sqrt(mean_squared_error(original_test, pred_test))
print('Test Score: %.2f RMSE' % (testScore1))
testScore2 = mean_absolute_error(original_test, pred_test)
print('Test Score: %.2f MAE' % (testScore2))
testScore3 = r2_score(original_test, pred_test)
print('Test Score: %.2f R2' % (testScore3))
testScore4 = mean_absolute_percentage_error(original_test, pred_test)
print('Test Score: %.2f MAPE' % (testScore4))

trainScore1 = math.sqrt(mean_squared_error(original_train, pred_train))
print('train Score: %.2f RMSE' % (trainScore1))
trainScore2 = mean_absolute_error(original_train, pred_train)
print('train Score: %.2f MAE' % (trainScore2))
trainScore3 = r2_score(original_train, pred_train)
print('train Score: %.2f R2' % (trainScore3))
trainScore4 = mean_absolute_percentage_error(original_train, pred_train)
print('train Score: %.2f MAPE' % (trainScore4))

df = pd.DataFrame({'Test Score: %.2f RMSE': [testScore1],
                   'Test Score: %.2f MAE': [testScore2],
                   'Test Score: %.2f R2': [testScore3],
                   'Test Score: %.2f MAPE': [testScore4],
                   'Train Score: %.2f RMSE': [trainScore1],
                   'Train Score: %.2f MAE': [trainScore2],
                   'Train Score: %.2f R2': [trainScore3],
                   'Train Score: %.2f MAPE': [trainScore4]
                   })

df.to_excel(os.path.join(base_dir, 'result', 'lstm.xlsx'), index=False)