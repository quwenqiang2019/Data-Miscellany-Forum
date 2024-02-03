import pandas as pd
import numpy as np

dates = pd.date_range(start='2015-01', end='2017-1', freq='M')
data = pd.Series([10, 12, 15, 200, 25, 30, 100, 22, 18, 15, 13, 11, 10, 12, 15, 20, 25, 300, 22, 18, 15, 13, 11, 10], index=dates)
print(data)
def replace_outliers(series):
    for date in series.index:
        year = date.year
        month = date.month
        if series[date] > 100:
            same_month_data = series[(series.index.year != year) & (series.index.month == month)]
            month_mean = same_month_data.mean()
            series[date] = month_mean
    return series

replaced_data = replace_outliers(data)
print(replaced_data)
