from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


# 构建模型的函数
def create_model(neurons_1):
    # 创建模型
    model = Sequential()
    model.add(Dense(neurons_1, input_shape=(8, ), kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 为了复现，设置随机种子
seed = 7
np.random.seed(seed)

# 加载数据
dataset = pd.read_csv("pima-indians-diabetes.csv", header=None)
dataset = pd.DataFrame(dataset)
print(dataset)
# 切分数据为输入 X 和输出 Y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]

# 创建模型，使用到了上一步找出的 epochs、batch size 最优参数
model = KerasClassifier(model=create_model, epochs=100, batch_size=80, verbose=0, neurons_1=1)
# 定义网格搜索参数
param_grid = {'neurons_1': [1, 5, 10, 15, 20, 25, 30]}
grid = GridSearchCV(estimator=model,  param_grid=param_grid)

grid_result = grid.fit(X, Y)
#
# 总结结果
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))