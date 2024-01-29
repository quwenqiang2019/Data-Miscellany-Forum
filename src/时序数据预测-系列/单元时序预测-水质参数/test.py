import pandas as pd
from scipy.interpolate import interp1d

# 创建一个包含不规律时间间隔的DataFrame对象
df = pd.DataFrame({
    'time': ['2013-01-01', '2013-02-01', '2013-04-01', '2013-06-01', '2013-07-01'],
    'value': [2, 4, 5, 7, 9]
})

# 将时间列转换为 pandas 的日期时间类型
df['time'] = pd.to_datetime(df['time'])

# 构造规律的时间间隔
start_time = df['time'].min()
end_time = df['time'].max()
regular_time = pd.date_range(start=start_time, end=end_time, freq='MS')

# 使用 interp1d 函数进行插值操作
interp_func = interp1d(df['time'].values.astype(float), df['value'].values)

# 对规律时间间隔进行插值操作
interpolated_values = interp_func(regular_time.values.astype(float))

# 创建包含插值结果的DataFrame对象
interpolated_df = pd.DataFrame({'time': regular_time, 'value': interpolated_values})

# 输出结果
print(interpolated_df)



import pandas as pd
from scipy.interpolate import interp1d

# 创建一个包含不规律时间间隔的DataFrame对象
df = pd.DataFrame({
    'time': ['2013-01-01', '2013-02-01', '2013-04-01', '2013-06-01', '2013-07-01'],
    'value': [2, 4, 5, 7, 9]
})

# 将时间列转换为 pandas 的日期时间类型
df['time'] = pd.to_datetime(df['time'])
# 将日期列设置为索引
df.set_index('time', inplace=True)
# 构造规律的时间间隔
df = df.resample('MS').asfreq()

# 使用插值法填充缺失值
df['value'] = df['value'].interpolate()
interpolated_df = df

# 输出结果
print(interpolated_df)