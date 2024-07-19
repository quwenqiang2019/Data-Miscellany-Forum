import tensorflow as tf
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input
from keras.models import  Sequential, Model
from keras.losses import MSE
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


# 1、加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 2、数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3、划分训练集和测试集，训练集120、测试集30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4、创建模型实例
input_size = X.shape[1]
hidden_size = 64
output_size = len(set(y))

# #  ========法1：使用Sequential API创建模型
# model = Sequential([
#     Input(shape=(input_size,)),
#     Dense(hidden_size, activation='relu'),
#     Dense(hidden_size, activation='relu'),
#     Dense(output_size, activation='softmax')
# ])

# =========法2：使用functional API创建模型
input = Input(shape=(input_size,))
hidden1 = Dense(hidden_size, activation='relu')(input)
hidden2 = Dense(hidden_size, activation='relu')(hidden1)
output = Dense(output_size, activation='softmax')(hidden2)
model = Model(inputs=input, outputs=output)

# # =========法3：使用subclassing API创建模型
# class NeuralNetwork(tf.keras.Model):
#     def __init__(self, hidden_size, output_size):
#         super(NeuralNetwork, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
#         self.dense3 = tf.keras.layers.Dense(output_size, activation='softmax')
#
#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         x = self.dense3(x)
#         return x
# model = NeuralNetwork(hidden_size, output_size)
# model.build(input_shape=(None, input_size))


model.summary()


# 5、训练模型
# =========法1：使用fit方法训练模型
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# ========法2：使用train_on_batch方法训练模型
train_db = tf.data.Dataset.from_tensor_slices((X_train, y_train))  # 构建Dataset对象
train_db = train_db.shuffle(100).batch(2)  # 随机散打，批量化过程
test_db = tf.data.Dataset.from_tensor_slices((X_test, y_test))  # 构建Dataset对象
test_db = test_db.shuffle(100).batch(2)  # 随机散打，批量化过程

optimizer = tf.keras.optimizers.Adam(0.001)
train_losses = []
test_losses = []
train_acc = []
test_acc = []



for epoch in range(10):
    for step, (train_x, train_y) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 梯度记录器
            out = model(train_x)  # 通过网络获得输出
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(train_y, out) # 创建一个 SparseCategoricalCrossentropy 实例计算损失

        if step % 10 == 0:
            print(epoch, step, float(loss))
        #  更新梯度信息
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_losses.append(float(loss))



    # 测试部分（每个epoch下均进行一次计算）
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0

    for test_x, test_y in test_db:
        y = model(test_x)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=test_y.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, test_y), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += test_x.shape[0]
        # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")



# 评估模型
y_pred = model.predict(X_test) #  返回numpy数组格式
y_pred = np.argmax(y_pred, axis=1)
print(y_pred, y_test)


acc = accuracy_score(y_test, y_pred) # 准确率acc
cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
cr = classification_report(y_test, y_pred) # 分类报告
print(acc, cm, cr, sep='\n')
