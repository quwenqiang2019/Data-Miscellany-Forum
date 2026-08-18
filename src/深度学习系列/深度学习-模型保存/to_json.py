import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential, model_from_json


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建模型实例
# input_size = X.shape[1]
# hidden_size = 64
# output_size = len(set(y))
#
# model = Sequential([
#     Dense(hidden_size, activation='relu', input_shape=[input_size]),
#     Dense(hidden_size, activation='relu'),
#     Dense(output_size, activation='softmax')
# ])
#
#
# model.summary()
#
# # 编译模型
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 训练模型
# model.fit(X_train, y_train, epochs=10, batch_size=32)
#
# # 保存模型的架构，生成json文件
# model_json = model.to_json()
# with open(r'model_json.json', 'w') as f:
#     f.write(model_json)
#     print('模型的架构json文件保存完成！')



# # #  =============================
# 加载模型的架构，模型配置和训练要保留，不需要保留模型的结构
with open(r'model_json.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(test_accuracy)