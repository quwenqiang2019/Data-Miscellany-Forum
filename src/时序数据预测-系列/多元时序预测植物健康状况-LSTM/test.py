from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy  as np

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


if __name__ == '__main__':

    elements=['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    input_data = DataFrame([[0, -16, -4.0, 1020.0, 2, 1.79, 0, 0]],
                        columns=elements)
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)

    dataset=dataset.append(input_data)
    values = dataset.values
    encoder = LabelEncoder()

    #字符转成数字编码
    #values[:,4] = encoder.fit_transform(values[:,4])
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)

    values = reframed.values
    test_X= values[:, :-1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(test_X.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(test_X.shape[1], test_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.load_weights("model.h5")

    y_predict = model.predict(np.array([test_X[-1]]))
    print(elements)
    print(input_data.values[0][1:])
    print("预测污染程度为："+str(y_predict[0][0]))