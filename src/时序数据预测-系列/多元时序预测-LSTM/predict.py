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
import requests
from lxml import etree
import json

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


def spider():
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    url = requests.get(
        "改成自己的系统。此处数据为爬取的数据进行预测，需要自行修改",
        headers=header)
    # 为了防止中文乱码，编码使用原网页编码
    url.raise_for_status()
    url.encoding = url.apparent_encoding
    text = url.text

    user_info = '{"data" }'
    user_dict = json.loads(text)
    # print(user_dict['data'])

    # 1号大棚
    # print(user_dict['data'][0]['areapoint'][1])
    temperature = user_dict['data'][0]['areapoint'][0]['value']  # 温度
    humidity = user_dict['data'][0]['areapoint'][1]['value']  # 湿度
    light_intensity = user_dict['data'][0]['areapoint'][2]['value']  # 光照强度
    soil_temperature = user_dict['data'][0]['areapoint'][3]['value']  # 土壤温度
    soil_humidity = user_dict['data'][0]['areapoint'][4]['value']  # 土壤湿度
    co2 = user_dict['data'][0]['areapoint'][5]['value']  # 二氧化碳
    rain = 1

    input_list = [0, temperature, humidity, light_intensity, soil_temperature, soil_humidity, co2, rain]
    input_list = [float(_) for _ in input_list]
    print(input_list)
    return input_list

if __name__ == '__main__':
    # load dataset
    dataset = read_csv('sate.csv', header=0, index_col=0)
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 15)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.load_weights("model.h5")


    while True:
        elements=['health','temperature','humidity','light_intensity','soil_temperature','soil_humidity','co2','rain']
        i=0
        if i%1==0:
            input_list=spider()
            input_data = DataFrame([input_list],
                                columns=elements)



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
        print(values.shape)
        test_X= values[:, :-1]
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print((test_X.shape))

        y_predict = model.predict(np.array([test_X[-1]]))
        print(elements)
        print(input_data.values[0][1:])
        print("预测污染程度为："+str(y_predict[0][0]))