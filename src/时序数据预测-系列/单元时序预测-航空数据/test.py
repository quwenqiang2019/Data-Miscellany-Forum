import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
import tqdm
# import statsmodels
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
df = pd.read_csv('international-airline-passengers.csv', encoding='utf-8')
print(df.head())
df.Month = pd.to_datetime(df.Month)  # 将字符串索引转换成时间索引
print(df.head())
ts = df['Passengers']  # 生成pd.Series对象
ts = ts.astype('float')
print(ts.head())