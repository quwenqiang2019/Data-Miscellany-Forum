from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = read_csv('sate.csv', header=0, index_col=0)
print(dataset.head())
print(dataset.shape)


test_split=round(len(dataset)*0.20)
train = dataset[:-test_split]
test = dataset[-test_split:]

print(train.shape)
print(test.shape)
# # 绘制训练集和测试集的折线图
# plt.figure(figsize=(10, 6))
# plt.plot(train, label='Training Data')
# plt.plot(test, label='Testing Data')
# plt.xlabel('Day')
# plt.ylabel('Open value')
# plt.title('Training and Testing Data')
# plt.legend()
# plt.show()



# 将数据归一化到 0~1 范围
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train)
test_scaled=scaler.transform(test)




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



Train = series_to_supervised(train_scaled, 3, 1)
print(Train.shape)
train_X, train_y = Train[:, :-1], Train[:, -1]