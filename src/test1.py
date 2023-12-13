import numpy as np
from sklearn.preprocessing import MinMaxScaler #导入库

data = np.random.randint(0,5,size=5) #随机生成长度为5的数据
data = np.array(data).reshape((len(data), 1))
print('原始数据：', data, sep='\n')
m = MinMaxScaler() #建立一个归一化器
data_1 = m.fit_transform(data) #利用m对data进行归一化，并储存data的归一化参数
print('归一化数据：', data_1, sep='\n')
data_2 = m.inverse_transform(data_1) #利用m对data_1进行反归一化
print('反归一化数据：', data_2, sep='\n')


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# 建立数据集:假设训练集有10个样本，测试集有5个样本；两个输入特征，一个输出
train_data = np.array([[0,1,3],[3,2,0],[0,2,3],[4,3,4],[3,0,1],[4,3,2],[2,3,3],[1,4,3],[0,4,4],[3,1,0]])
train_data = pd.DataFrame(train_data, columns = ['output','input1','input2'])
x_train = train_data[['input1','input2']].values
y_train = train_data[['output']].values

test_data = np.random.randint(5,size=(5,3))
test_data = pd.DataFrame(test_data ,columns = ['output','input1','input2'])
x_test = test_data[['input1','input2']].values
y_test = test_data[['output']].values

# 对训练集进行归一化,特征和标签可以分开归一化处理，也可以一起，效果都是一样的，如果一起后面反归一化预测值会麻烦些
# mm = MinMaxScaler()    # 特征和标签一起归一化处理
# train_data_m = mm.fit_transform(train_data)
mm1 = MinMaxScaler()   # 特征进行归一化
x_train_m = mm1.fit_transform(x_train)
mm2 = MinMaxScaler()     # 标签进行归一化
y_train_m = mm2.fit_transform(y_train)

# 将归一化的训练数据输入模型，经过模型训练，得到了模型model
Model = LinearRegression()
Model.fit(x_train_m, y_train_m)

# 对测试集特征进行相同规则mm1的归一化处理，然后输入到模型进行预测
x_test_m = mm1.transform(x_test) #注意fit_transform() 和 transform()的区别
predicted_y_m = Model.predict(x_test_m) #利用输入特征input1和input2测试模型

# 预测结果进行相同规则mm2反归一化
predicted_y = mm2.inverse_transform(predicted_y_m)
print(predicted_y)

