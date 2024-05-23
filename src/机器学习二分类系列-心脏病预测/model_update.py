import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df)
# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
x = df[features].values
y = df[target].values

# x = load_iris().data
# y = load_iris().target
# print(x)
# print(y)

x_disorder, y_disorder = shuffle(x, y, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x_disorder, y_disorder, random_state=3)

x_old_train = x_train[:90, :]
y_old_train = y_train[:90]
x_new_train = x_train[90:, :]
y_new_train = y_train[90:]

# 先在一部分数据上fit训练，然后在此训练好模型的基础上继续fit
model = RandomForestClassifier()
model = model.fit(x_old_train, y_old_train)  # 先用一部分训练数据，训练模型，并保存
with open('old_version.pickle', 'wb') as f:
    pickle.dump(model, f)
# 调用刚刚保存的模型，比如此时又来新的训练数据，这时模型继续训练，再预测
pickle_in = open('old_version.pickle', 'rb')
model = pickle.load(pickle_in)
model = model.fit(x_new_train, y_new_train)



# 直接训练所有数据后，再预测的结果
all_model = RandomForestClassifier()
all_model = all_model.fit(x_train, y_train)


# 两种结果存在一定的偏差，但是这样的好处是：模型参数不需要重新训练，只需要再以前的基础上继续训练节省时间。
print('基于已有模型+新训练数据得到模型结果', model.score(x_test, y_test))
print('训练所有数据得到的结果', all_model.score(x_test, y_test))