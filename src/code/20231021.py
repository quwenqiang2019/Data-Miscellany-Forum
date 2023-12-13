import pandas as pd
import numpy as np

# 创建一个示例DataFrame
data = {'A': ['apple', 'banana', 'apple', 'orange', 'grape']}
df = pd.DataFrame(data)

# 打印替换前的DataFrame
print("替换前的DataFrame：")
print(df)

# 指定要替换为0的字符串
string_0 = 'apple'
# 指定要替换为1的字符串
string_1 = 'banana'

# 使用numpy.where函数进行条件替换
df['A'] = np.where(df['A'] == string_0, 0, np.where(df['A'] == string_1, 1, 2))

# 打印替换后的DataFrame
print("替换后的DataFrame：")
print(df)