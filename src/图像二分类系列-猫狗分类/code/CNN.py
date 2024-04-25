import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

current_dir = 'E:\data\cat_vs_dog'
print(current_dir)

# 创建新的目录来存储需要的数据集
base_dir = current_dir + '/cat_vs_dog_small'
if  not os.path.exists(base_dir):
    os.mkdir(base_dir)

# 分别创建训练集、验证集的目录
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

# 创建猫、狗的训练、验证图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
if  not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

# # cat数据集复制
# fnames = ['{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(current_dir, 'Cat', fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(current_dir, 'Cat', fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# # dog数据集复制
# fnames = ['{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(current_dir, 'Dog', fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(current_dir, 'Dog', fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)



# 数据预处理
train_datagen = ImageDataGenerator(rescale=1./255)  # 进行缩放
test_datagen = ImageDataGenerator(rescale=1./255)  # 进行缩放

train_generator = train_datagen.flow_from_directory(
    train_dir,  # 待处理的目录
    target_size=(150,150),  # 图像大小设置
    batch_size=20,
    class_mode="binary"  # 损失函数是binary_crossentropy 所以使用二进制标签
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,  # 待处理的目录
    target_size=(150,150),  # 图像大小设置
    batch_size=20,
    class_mode="binary"  # 损失函数是binary_crossentropy 所以使用二进制标签
)

for data_batch, labels_batch in train_generator:
    print(data_batch.shape)
    print(labels_batch.shape)
    break

#  构建模型
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",
                               input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))  #

model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy",
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=["acc"])


history = model.fit_generator(
    train_generator,  # 第一个参数必须是Python生成器
    steps_per_epoch=100,  # 2000 / 20
    epochs=30,  # 迭代次数
    validation_data=validation_generator,  # 待验证的数据集
    validation_steps=50
)

model.save("cats_and_dogs_small.h5")

# 绘制训练过程中的损失和准确率
history_dict = history.history  # 字典形式
acc = history_dict["acc"]
val_acc = history_dict["val_acc"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc)+1)
# acc
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Validation acc")
plt.legend()
plt.show()

# loss
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.show()