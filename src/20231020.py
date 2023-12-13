import pandas as pd
import re

# 创建一个示例DataFrame
data = {'A': ['apple', 'banana', 'pineapple', 'orange', 'grape']}
df = pd.DataFrame(data)

# 打印替换前的DataFrame
print("替换前的DataFrame：")
print(df)

# 定义要进行模糊匹配的字符串
pattern = r'^ap'

# 定义要替换的值
replacement = 'fruit'

# 使用正则表达式进行模糊匹配替换
df['A'] = df['A'].replace(to_replace=pattern, value=replacement, regex=True)

# 打印替换后的DataFrame
print("替换后的DataFrame：")
print(df)