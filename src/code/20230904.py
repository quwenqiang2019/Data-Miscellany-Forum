# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据，因为原始数据是顺序的，顺序不打乱会影响准确率
# seed：随机种子，是一个整数，当设置之后，每次生成的随机数都一样
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱的数据集分为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后界矩阵相乘会因数据类型不一样报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签一一对应（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数，4个输入特征，故输入层为4个节点，，因为分为3类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续话acc曲线提供数据
epoch = 500  # 循环500次
loss_all = 0  # 每轮分为4个step，loss_all记录四个step生成的4个loss和

# 训练
for epoch in range(epoch):  # 数据级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络的乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热编码同量级）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热编码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean（sum（y - out）^2）
            loss_all += loss.numpy()  # 将每个step计算出来的loss值累加，为后续求loss平均值提供数据，这样计算的loss更准确
        grads = tape.gradient(loss, [w1, b1])  # 对loss函数进行求梯度
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b1自更新
    print("Epoch {},loss: {}".format(epoch, loss_all / 4))  # 每个epoch，打印loss信息
    train_loss_results.append(loss_all / 4)  # 将4个step的loss平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录一个epoch的loss做准备

    # 测试
    # total_correct为预测对的样本的个数，total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)  # 将pred转换为y_test的数据类型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)  # 若分类正确，则correct=1，否则correct=0，将bool型的结果转换为int型
        correct = tf.reduce_sum(correct)  # 将每个batch的correct数加起来
        total_correct += int(correct)  # 将所有batch的correct数加起来
        total_number += x_test.shape[0]  # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
    acc = total_correct / total_number  # 总的正确率等于total_correct / total_number
    test_acc.append(acc)
    print("Test_acc", acc)
    print("--------------------")

# 绘制loss曲线
plt.title('loss Function Curve')  # 图片标题
plt.xlabel('epoch')  # x轴变量名称
plt.ylabel('loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出train_loss_results值并连线，连接图标是loss
plt.legend()  # 画出曲线坐标
plt.show()  # 画出图像

# 绘制Accuracy曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连接图标是Accuracy
plt.legend()
plt.show()
