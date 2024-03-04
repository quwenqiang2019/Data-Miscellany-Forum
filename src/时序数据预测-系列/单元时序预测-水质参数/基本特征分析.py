import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt


# 读取数据集
data = pd.read_excel('5样点 - 副本 - 副本.xlsx')
data = pd.DataFrame(data)
# 将日期列转换为日期时间类型
data['日期'] = pd.to_datetime(data['日期'], format='%Y%m', infer_datetime_format=True)
data = data[['日期', 'TSMvalue']]
# 将日期列设置为索引
data.set_index('日期', inplace=True)


# 时序数据分解
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data)
result.plot()
plt.show()

# ACF：自相关函数
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data).show()
plt.show()

# PACF：偏自相关函数
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data).show()
plt.show()

# 平稳性检验：Dickey-Fuller检验
from statsmodels.tsa.stattools import adfuller
adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(data)
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF used number of lags:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion: ', icbest)