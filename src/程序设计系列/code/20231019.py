import pandas as pd

# 创建一个示例DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# 打印替换前的DataFrame
print("替换前的DataFrame：")
print(df)

# 指定要替换为0的值
value_0 = 'a'
# 指定要替换为1的值
value_1 = 'c'

# 使用replace方法结合条件语句替换列'B'中的值
df['B'] = df['B'].replace({value_0: 0, value_1: 1})

# 打印替换后的DataFrame
print("替换后的DataFrame：")
print(df)