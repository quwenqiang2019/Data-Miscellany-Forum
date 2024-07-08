import os
import pandas as pd
import numpy as np
import random
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 去掉不必要的报错

# 老规矩，为了可以复现结果，指定一下随机数种子
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)


# 利用pandas读取数据集
# 字段有效能（公里数每加仑），气缸数，排量，马力，重量，加速度，型号年份，产地
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv('auto-mpg.data',  names=['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin'], na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

# 查看部分数据
dataset.head()
dataset.isna().sum()  # 统计空白数据
dataset = dataset.dropna()  # 删除空白数据项
dataset.isna().sum()  # 再次统计

# 处理类别型数据，其中 origin 列代表了类别 1,2,3,分布代表产地：美国、欧洲、日本
# 先弹出(删除并返回)origin 这一列
origin = dataset.pop('Origin')
# 根据 origin 列来写入新的 3 个列
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()  # 查看新表格的最后几项

# 数据集 = 训练集 + 测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 统计数据
sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]],
             diag_kind="kde")

# 查看训练集的输入x的统计数据
train_stats = train_dataset.describe()
train_stats.pop("MPG")  # 仅保留输入x
train_stats = train_stats.transpose()  # 转置

# 移动MPG油耗效能这一列为真实标签y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 标准化数据
def norm(x):
    # 减去每个字段的均值，并除以标准差
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)  # 标准化数据集
normed_test_data = norm(test_dataset)  # 标准化测试集

print(normed_train_data.shape, train_labels.shape)
print(normed_test_data.shape, test_labels.shape)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')



EPOCHS = 1000
model = build_model()
history = model.fit(normed_train_data, train_labels,epochs=EPOCHS, validation_split=0.2, verbose=0,callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n', hist.tail())
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [MPG]')
plt.plot(hist['epoch'], hist['mae'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mae'],
         label='Val Error')
plt.ylim([0, 5])
plt.legend()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(hist['epoch'], hist['mse'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mse'],
         label='Val Error')
plt.ylim([0, 20])
plt.legend()
plt.show()


# patience 值用来检查改进 epochs 的数量
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])


# 参数：
# filename：字符串，保存模型的路径
# monitor：需要监视的值
# verbose：信息展示模式，0或1
# save_best_only：当设置为True时，将只保存在验证集上性能最好的模型

filepath = "model_{epoch:02d}-{val_mse:.2f}.h5"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
    period=3
)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,validation_split=0.2, verbose=0, callbacks=[checkpoint, PrintDot()])




# 模型评估
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# 误差分布
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")