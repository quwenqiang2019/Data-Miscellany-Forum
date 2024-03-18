import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
df = pd.read_csv(os.path.join(base_dir, 'data', "螺纹钢加权.csv"), encoding = 'gb2312')

df = pd.DataFrame(df)
df = df.iloc[:, 0:9]
# 合并日期和时间列为一个DateTime列
df['DateTime'] = pd.to_datetime(df['日期'] + ' ' + df['时间'])
# 删除日期和时间两列
df.drop(['日期', '时间'], axis=1, inplace=True)
df.set_index('DateTime', inplace = True)
print(df.head())
df.insert(6, '收盘', df.pop('收盘'))
print(df.shape)
print(df.head())


test_split=round(len(df)*0.20)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]


# 绘制训练集和测试集的折线图
sns.set_style('darkgrid')
font1 = {'family': ['Times New Roman','SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6))
plt.plot(df_for_training['收盘'], label='Training Data')
plt.plot(df_for_testing['收盘'], label='Testing Data')
plt.xlabel('时间')
plt.xticks(rotation=45)
plt.ylabel('收盘价')
plt.title('训练集和测试集')
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

window_size = 30
trainX,trainY=createXY(df_for_training_scaled,window_size)
testX,testY=createXY(df_for_testing_scaled,window_size)
print(trainY[0])
print(trainY[0])

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, 7))
testX = np.reshape(testX, (testX.shape[0], window_size, 7))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)


def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,7)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
grid_search = grid_search.fit(trainX,trainY)
print(grid_search.best_params_)
my_model=grid_search.best_estimator_.model




prediction=my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)



prediction_copies_array = np.repeat(prediction,7, axis=-1)
print(prediction_copies_array.shape)
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),7)))[:,0]
original_copies_array = np.repeat(testY, 7, axis=-1)
print(original_copies_array.shape)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),7)))[:,0]
print("Pred Values-- ", pred)
print("\nOriginal Values-- ", original)


plt.plot(df_for_testing.index[window_size:,], original, color = 'red', label = '真实值')
plt.plot(df_for_testing.index[window_size:,], pred, color = 'blue', label = '预测值')
plt.title('收盘价预测')
plt.xlabel('时间')
plt.xticks(rotation=45)
plt.ylabel('收盘价')
plt.legend()
plt.savefig(os.path.join(base_dir, 'result', 'lstm_pred.jpg'), bbox_inches='tight', dpi = 600)
plt.show()

# 计算误差
writer = pd.ExcelWriter(os.path.join(base_dir, 'result', '测试集评价指标.xlsx'))
testScore1 = math.sqrt(mean_squared_error(original, pred))
print('Test Score: %.2f RMSE' % (testScore1))

testScore2 = mean_absolute_error(original, pred)
print('Test Score: %.2f MAE' % (testScore2))

testScore3 = r2_score(original, pred)
print('Test Score: %.2f R2' % (testScore3))

testScore4 = mean_absolute_percentage_error(original, pred)
print('Test Score: %.2f MAPE' % (testScore4))

df = pd.DataFrame({'Test Score: %.2f RMSE': [testScore1],
                   'Test Score: %.2f MAE': [testScore2],
                   'Train Score: %.2f R2': [testScore3],
                   'Train Score: %.2f MAPE': [testScore4]})

df.to_excel(writer, sheet_name='lstm', index=False)
writer.save()