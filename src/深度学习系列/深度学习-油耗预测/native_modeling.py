# 1. 数据集
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 利用pandas读取数据集
raw_dataset = pd.read_csv('mpg.csv')
raw_dataset = raw_dataset.iloc[:,:-1]
dataset = raw_dataset.copy()
print(dataset.head())
dataset = dataset.dropna()  # 删除空白数据

# 将Origin的3个产地分别用1，2，3代替
origin = dataset.pop('origin')
dataset['usa'] = (origin == 'usa') * 1.0
dataset['europe'] = (origin == 'europe') * 1.0
dataset['japan'] = (origin == 'japan') * 1.0

# 8:2划分数据集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 移动MPG油耗效能这列数据为真实标签y
train_lables = train_dataset.pop('mpg')
test_lables = test_dataset.pop('mpg')

# 查看训练集X的数据
train_status = train_dataset.describe()
train_status = train_status.transpose()


# 标准化数据
def norm(x):
    return (x - train_status['mean']) / (train_status['std'])


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 打印训练集和测试机的大小
print(normed_train_data.shape, train_lables.shape)
print(normed_test_data.shape, test_lables.shape)

# 利用切分的训练集数据构建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_lables.values))  # 构建Dataset对象
train_db = train_db.shuffle(100).batch(32)  # 随机散打，批量化过程


# 2.创建网络，这里采用子类化模型构建，也可以通过keras.Sequential()构建或者函数式模型构建
class Network(keras.Model):
    # 回归网络
    def __init__(self):
        super(Network, self).__init__()
        # 创建3个全连接层
        self.fc1 = keras.layers.Dense(64, activation="relu")
        self.fc2 = keras.layers.Dense(64, activation="relu")
        self.fc3 = keras.layers.Dense(1)

    def call(self, input, training=None, mask=None):
        # 依次通过3个连接层
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


# 3. 训练与测试：实例化网络对象和创建优化器，这里原生编译和训练过程，这里train_db是一个Dataset对象，必须经过转化，也可以通过model.fit()方法进行训练
model = Network()
model.build(input_shape=(None, 9))
model.summary()

optimizer = tf.keras.optimizers.RMSprop(0.001)
train_mae_losses = []
test_mae_losses = []

for epoch in range(200):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 梯度记录器
            out = model(x)  # 通过网络获得输出
            loss = tf.reduce_mean(keras.losses.MSE(y, out))  # 计算MSE
            mae_loss = tf.reduce_mean(keras.losses.MAE(y, out))  # 计算MAE

        if step % 10 == 0:
            print(epoch, step, float(loss))
        #  更新梯度信息
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_mae_losses.append(tf.reduce_mean(keras.losses.MAE(test_lables, out)))


plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_mae_losses, label='Train')

plt.plot(test_mae_losses, label='Test')
plt.legend()

# plt.ylim([0,10])
plt.legend()
plt.savefig('auto.svg')
plt.show()
