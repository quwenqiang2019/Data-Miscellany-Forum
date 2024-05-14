import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Bidirectional, Input
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
df = pd.read_csv(os.path.join(base_dir, 'data', "沪深300.csv"), encoding = 'gb2312')
df = pd.DataFrame(df)
df = df.iloc[:, 0:26]   # 选择需要的列，这里选择了前8列，如果增加了其他变量这里需要做改动
df = df.dropna()
target = '收盘'
df['DateTime'] = pd.to_datetime(df['日期'])  # 将日期列转换为日期格式，并新增一列DateTime
df.drop(['日期'], axis=1, inplace=True)  # 删除日期列
df.set_index('DateTime', inplace = True) # 将DateTime列设置为索引
df.insert(0, target, df.pop(target)) # 由于开盘是需要预测的值，这里需要将开盘列移动到第一列
fea_num = len(df.columns)  # 计算一下数据的变量数量，也就是列数
print(df)

sns.set(font_scale=1.2)
plt.rc('font',family=['SimSun'], size=12)
plt.figure(figsize=(20, 16))
plt.subplots_adjust()
ax = sns.heatmap(df.corr(), annot=True, xticklabels=False, fmt=".2f")
ax.set_title('相关性热力图')  # 图标题
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'result', 'heatmap.jpg'), bbox_inches='tight', dpi=1600)
plt.show()


test_split = round(len(df)*0.20)   # 设置测试集的大小，这里设置为总数据量的20%
df_for_training=df[:-test_split]  #  划分训练集和测试集
df_for_testing=df[-test_split:]


# 绘制训练集和测试集的折线图，这里只绘制开盘这个特征
sns.set_style('darkgrid')
font1 = {'family': ['SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6))
plt.plot(df_for_training[target], label='Training Data')
plt.plot(df_for_testing[target], label='Testing Data')
plt.xlabel('时间')
plt.xticks(rotation=45)
plt.ylabel(target)
plt.title(target)
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


def build_model(neurons_1, neurons_2, dropout_rate):   # 这里采用sequential模型，也可以采用其他模型进行网络的搭建,模型的结构可以自行做调整
    grid_model = Sequential()
    grid_model.add(Input(shape=(window_size, fea_num)))
    grid_model.add(Bidirectional(LSTM(neurons_1, return_sequences=True)))
    grid_model.add(Dropout(dropout_rate))
    grid_model.add(Bidirectional(LSTM(neurons_2)))
    grid_model.add(Dropout(dropout_rate)) # 这一层为了防止过拟合
    grid_model.add(Dense(1))

    grid_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # 对模型进行编译，一般需要指定损失函数loss、优化器optimizer和评估标准metrics
    return grid_model

grid_model = KerasRegressor(model=build_model, verbose=1, neurons_1=50, neurons_2=50, dropout_rate=0.2)  # 使用 Keras 中的 KerasRegressor 类来封装一个自定义的回归模型。使用了一个名为 build_model 的函数或模型作为参数传递给 KerasRegressor
parameters = {
              'batch_size': [16],
              'epochs': [8],
              'neurons_1': [50],
              'neurons_2': [50],
              'dropout_rate': [0.0, 0.1, 0.2, 0.4, 0.5]
              }    # parameters 是一个字典，其中包含了需要调优的超参数及其对应的备选取值。在这个例子中，'batch_size' 表示批量大小，备选取值为 [16, 20]；'epochs' 表示训练轮数，备选取值为 [8, 10]；'optimizer' 表示优化器的选择，备选取值为 ['adam', 'Adadelta']

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
plt.savefig(os.path.join(base_dir, 'result', 'lstm_pred_train.jpg'), bbox_inches='tight', dpi = 600)
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
plt.savefig(os.path.join(base_dir, 'result', 'lstm_pred_test.jpg'), bbox_inches='tight', dpi = 600)
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


index = ['Test Score: %.2f RMSE',
          'Test Score: %.2f MAE',
          'Test Score: %.2f R2',
          'Test Score: %.2f MAPE',
          'Train Score: %.2f RMSE',
          'Train Score: %.2f MAE',
          'Train Score: %.2f R2',
          'Train Score: %.2f MAPE']

values = [testScore1,  testScore2, testScore3, testScore4, trainScore1, trainScore2, trainScore3, trainScore4]
df = pd.DataFrame({'index': index, 'values': values})
print(df)

df.to_excel(os.path.join(base_dir, 'result', 'lstm.xlsx'), index=False)   # 将评估指标值存为数据表

values1 = [round(trainScore1, 2),  round(trainScore2, 2), round(trainScore3, 2), round(trainScore4, 2)]
values2 = [round(testScore1, 2),  round(testScore2, 2), round(testScore3, 2), round(testScore4, 2)]

values1_gy = [round(trainScore1, 2)/max(round(trainScore1, 2), round(testScore1, 2)),  round(trainScore2, 2)/max(round(trainScore2, 2), round(testScore2, 2)), round(trainScore3, 2)/max(round(trainScore3, 2), round(testScore3, 2)), round(trainScore4, 2)/max(round(trainScore4, 2), round(testScore4, 2))]
values2_gy = [round(testScore1, 2)/max(round(trainScore1, 2), round(testScore1, 2)),  round(testScore2, 2)/max(round(trainScore2, 2), round(testScore2, 2)), round(testScore3, 2)/max(round(trainScore3, 2), round(testScore3, 2)), round(testScore4, 2)/max(round(trainScore4, 2), round(testScore4, 2))]

def plot_radar(values1, values2):
    font = {'family': 'Times New Roman',
            'size': 12,
            }
    sns.set(font_scale=1.2)
    plt.rc('font', family='Times New Roman')
    plt.style.use('ggplot')  # 使用ggplot的绘图风格

    # 构造数据
    feature = ["RMSE", "MAE", "R2", "MAPE"]


    # 设置每个数据点的显示位置，在雷达图上用角度表示
    angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    feature = np.concatenate((feature, [feature[0]]))

    # 绘图
    fig = plt.figure(figsize=(8, 8))
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)

    for values in [values1, values2]:
        # 拼接数据首尾，使图形中线条封闭
        values = np.concatenate((values, [values[0]]))
        # 绘制折线图
        ax.plot(angles, values, 'o-', linewidth=2)

    for values in [values1, values2]:
        values = np.concatenate((values, [values[0]]))
        # 填充颜色
        ax.fill(angles, values, alpha=0.25)

    # 设置图标上的角度划分刻度，为每个数据点处添加标签
    ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=14, style='italic')
    # 设置雷达图的范围
    ax.set_ylim(0.1, 1)
    # 设置雷达图的0度起始位置
    ax.set_theta_zero_location('N')
    # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
    ax.set_rlabel_position(270)
    plt.legend(["train", "test"], loc='best')
    # 添加标题
    plt.title('Comparison of evaluation indicators', fontsize=14)
    # 添加网格线
    plt.savefig(os.path.join(base_dir, 'result', 'Comparison_radar.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

def plot_bar(values1, values2):
    font = {'family': 'Times New Roman',
            'size': 12,
            }
    sns.set(font_scale=1.2)

    features = ["RMSE", "MAE", "R2", "MAPE"]

    train = values1
    test = values2

    x = np.arange(len(features))
    width = 0.2
    train_x = x
    test_x = x + width

    # 绘图
    plt.bar(train_x, train, width=width, color='gold', label='train data')
    plt.bar(test_x, test, width=width, color="silver", label="test data")

    plt.xticks(x + width, labels=features)

    # 显示柱状图的高度文本
    for i in range(len(features)):
        plt.text(train_x[i], train[i], train[i], va="bottom", ha="center", fontsize=8)
        plt.text(test_x[i], test[i], test[i], va="bottom", ha="center", fontsize=8)

    # 显示图例
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(base_dir, 'result', 'Comparison_bar.jpg'), bbox_inches='tight', dpi=600)
    plt.show()




plot_bar(values1, values2)
plot_radar(values1_gy, values2_gy)