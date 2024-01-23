import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = read_csv('sate.csv', header=0, index_col=0)
df = pd.DataFrame(dataset)
print(dataset.head())
print(dataset.shape)


# 拆分数据集为训练集和测试集
test_split=round(len(df)*0.20)
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]




# 将数据归一化到 0~1 范围
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)




# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    #这一步报错，把所有nan变为0
    if dropnan:
        agg=agg.fillna(0)
    return agg



train = series_to_supervised(df_for_training, 2, 2)
# train_X, train_y = train[:, :-1], train[:, -1]
print(train.shape)
print(train.head())


# test = series_to_supervised(df_for_testing, 1, 1)
# # test_X, test_y = test[:, :-1], test[:, -1]
# print(test.shape)
# print(test.head())