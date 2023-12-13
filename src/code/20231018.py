import pandas as pd

# 创建一个示例DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# 打印映射替换前的DataFrame
print("映射替换前的DataFrame：")
print(df)

# 定义映射关系的字典
mapping = {'a': 'apple', 'b': 'banana', 'c': 'cherry'}

# 使用map方法根据列'B'的值进行映射替换
df['B'] = df['B'].map(mapping)

# 打印映射替换后的DataFrame
print("映射替换后的DataFrame：")
print(df)