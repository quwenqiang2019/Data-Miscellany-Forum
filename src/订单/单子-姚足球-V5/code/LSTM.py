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
df = pd.read_csv(os.path.join(base_dir, 'data', "300股指加权(日线).csv"), encoding = 'gb2312')
df = pd.DataFrame(df)
df = df.iloc[:, 0:8]   # 选择需要的列，这里选择了前8列，如果增加了其他变量这里需要做改动
df['DateTime'] = pd.to_datetime(df['日期'])  # 将日期列转换为日期格式，并新增一列DateTime
df.drop(['日期'], axis=1, inplace=True)  # 删除日期列
df.set_index('DateTime', inplace = True) # 将DateTime列设置为索引
df.insert(0, '开盘', df.pop('开盘')) # 由于开盘是需要预测的值，这里需要将开盘列移动到第一列
fea_num = len(df.columns)  # 计算一下数据的变量数量，也就是列数

test_split = round(len(df)*0.20)   # 设置测试集的大小，这里设置为总数据量的20%
df_for_training=df[:-test_split]  #  划分训练集和测试集
df_for_testing=df[-test_split:]


# 绘制训练集和测试集的折线图，这里只绘制开盘这个特征
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

scaler = MinMaxScaler(feature_range=(0,1))    # 定义归一化对象
df_for_training_scaled = scaler.fit_transform(df_for_training)  # 对训练集进行归一化
df_for_testing_scaled=scaler.transform(df_for_testing) # 对测试集采用同样的归一化参数，这里注意是transform不是fit_transform


def createXY(dataset,n_past):     # 这个函数是为了将时序数据转化为监督学习数据
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

window_size = 1  # 设置滑动窗口大小，这里可以做调整
trainX,trainY=createXY(df_for_training_scaled,window_size)  # 经过这个函数处理后，trainX就算特征，trainY就是目标值
testX,testY=createXY(df_for_testing_scaled,window_size) # 经过这个函数处理后，testX就算特征，testY就是目标值

# 将数据集转换为 LSTM 模型所需的形状（样本数，时间步长，特征数）
trainX = np.reshape(trainX, (trainX.shape[0], window_size, fea_num))
testX = np.reshape(testX, (testX.shape[0], window_size, fea_num))

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)


def build_model(neurons_1=50, neurons_2=50):   # 这里采用sequential模型，也可以采用其他模型进行网络的搭建,模型的结构可以自行做调整
    grid_model = Sequential()
    grid_model.add(LSTM(neurons_1,return_sequences=True,input_shape=(window_size,fea_num)))
    grid_model.add(LSTM(neurons_2))
    grid_model.add(Dropout(0.2)) # 这一层为了防止过拟合
    grid_model.add(Dense(1))

    grid_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # 对模型进行编译，一般需要指定损失函数loss、优化器optimizer和评估标准metrics
    return grid_model

grid_model = KerasRegressor(model=build_model,verbose=1)  # 使用 Keras 中的 KerasRegressor 类来封装一个自定义的回归模型。使用了一个名为 build_model 的函数或模型作为参数传递给 KerasRegressor
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'],
              'model__neurons_1': [50, 128, 256],
              'model__neurons_2': [50, 128, 256]}    # parameters 是一个字典，其中包含了需要调优的超参数及其对应的备选取值。在这个例子中，'batch_size' 表示批量大小，备选取值为 [16, 20]；'epochs' 表示训练轮数，备选取值为 [8, 10]；'optimizer' 表示优化器的选择，备选取值为 ['adam', 'Adadelta']

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)  # GridSearchCV 类用于系统地搜索指定参数值的组合，以找到最优的参数组合。在这里，estimator = grid_model 指定了要使用的估计器（即前面创建的 grid_model 对象），param_grid = parameters 指定了要搜索的参数网格，cv = 2 指定了交叉验证的折数为 2
grid_search = grid_search.fit(trainX,trainY)  # 进行训练拟合
print(grid_search.best_params_)  #  打印最佳的参数组合
my_model=grid_search.best_estimator_  # 确定最优的模型


prediction_test=my_model.predict(testX)  # 利用最优模型进行预测
prediction_train=my_model.predict(trainX)

prediction_train_copies_array = np.repeat(prediction_train,fea_num, axis=-1)
pred_train=scaler.inverse_transform(np.reshape(prediction_train_copies_array,(len(prediction_train),fea_num)))[:,0]    # 对预测值进行反归一化处理，首先通过 np.reshape 将 prediction_train_copies_array 重新塑形为原始形状，然后使用 scaler.inverse_transform 对其进行逆标准化操作，最后取出第一列数据，得到了原始数据的预测结果 pred_train
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
pred_test=scaler.inverse_transform(np.reshape(prediction_test_copies_array,(len(prediction_test),fea_num)))[:,0]  # 对测试集做同样的处理
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


df0 = pd.DataFrame({'日期': df_for_testing.index[window_size:],'真实值': original_test, '预测值': pred_test})
print(df0)
# 计算预测股价和前一天真实股价的比较结果
df0['比较结果'] = (df0['预测值'] > df0['真实值'].shift(1)).astype(int)
# 填充第一行的比较结果为 0
df.at[0, '比较结果'] = None
print(df0)
df0.to_excel(os.path.join(base_dir, 'result', '预测值对比前一天真实值.xlsx'), index=False)   # 将比较结果存为数据表


# 计算误差
testScore1 = math.sqrt(mean_squared_error(original_test, pred_test))  # 计算测试集RMSE
print('Test Score: %.2f RMSE' % (testScore1))
testScore2 = mean_absolute_error(original_test, pred_test) #  计算测试集MAE
print('Test Score: %.2f MAE' % (testScore2))
testScore3 = r2_score(original_test, pred_test) #   计算测试集R2
print('Test Score: %.2f R2' % (testScore3))
testScore4 = mean_absolute_percentage_error(original_test, pred_test) # 计算测试集MAPE
print('Test Score: %.2f MAPE' % (testScore4))

trainScore1 = math.sqrt(mean_squared_error(original_train, pred_train))  # 计算测试集RMSE
print('train Score: %.2f RMSE' % (trainScore1))
trainScore2 = mean_absolute_error(original_train, pred_train)   # 计算测试集MAE
print('train Score: %.2f MAE' % (trainScore2))
trainScore3 = r2_score(original_train, pred_train)  # 计算测试集R2
print('train Score: %.2f R2' % (trainScore3))
trainScore4 = mean_absolute_percentage_error(original_train, pred_train)  # 计算测试集MAPE
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

df.to_excel(os.path.join(base_dir, 'result', 'lstm.xlsx'), index=False)   # 将评估指标值存为数据表