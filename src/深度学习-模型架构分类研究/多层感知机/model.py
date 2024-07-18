import tensorflow as tf
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input
from keras.models import  Sequential, Model
from keras.losses import MSE
import keras




# 1、加载鸢尾花数据集
# iris = load_iris()
# X = iris.data
# y = iris.target

boston = load_boston()
X = boston.data
y = boston.target

# 2、数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3、划分训练集和测试集
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
train_db = train_db.shuffle(100).batch(32)  # 随机散打，批量化过程
optimizer = tf.keras.optimizers.RMSprop(0.001)
train_mae_losses = []
test_mae_losses = []

for epoch in range(200):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 梯度记录器
            out = model(x)  # 通过网络获得输出
            print(out)
            print(y)
            loss = tf.reduce_mean(keras.losses.MSE(y, out))  # 计算MSE
            mae_loss = tf.reduce_mean(keras.losses.MAE(y, out))  # 计算MAE

        if step % 10 == 0:
            print(epoch, step, float(loss))
        #  更新梯度信息
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(X_test))
    test_mae_losses.append(tf.reduce_mean(keras.losses.MAE(y_test, out)))


# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(test_accuracy)