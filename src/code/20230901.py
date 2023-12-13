import pandas as pd

df1 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data1': range(3)})
print(df1)

df2 = pd.DataFrame({'key': ['a', 'b', 'c', 'a', 'b'],
                    'data2': range(5)})
print(df2)

# 左连接
# df3 = pd.merge(df1, df2, how='left')
# 右连接
# df3 = pd.merge(df1, df2, how='right')
# 内连接:链接键为都存在的“key”列，结果为两个数据集都有的key列元素
# df3 = pd.merge(df1, df2, how='inner')
# 外连接
df3 = pd.merge(df1, df2, how='outer')
print(df3)