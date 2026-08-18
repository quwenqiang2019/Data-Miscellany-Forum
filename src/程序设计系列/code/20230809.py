import pandas as pd

# 创建dataframe
data = {'Name':['Tom1', 'Jack1', 'Steve1', 'Ricky1', 'Tom2', 'Jack2', 'Steve2', 'Ricky2'],'Score':[78,60,59,42,88,34,69,142]}
df = pd.DataFrame(data)
print(df)

# 定义区间和标签
bins = [0, 60, 80, 90, float('inf')]
labels = ['<=60', '60-80', '80-90', '90+']

# 将 Score 列的值分入不同区间，并计算频数
counts = pd.cut(df['Score'], bins=bins, labels=labels).value_counts().sort_index()

print(counts)