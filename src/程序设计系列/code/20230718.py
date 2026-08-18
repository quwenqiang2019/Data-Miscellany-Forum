import pandas as pd

# 创创建series
series= pd.Series([1, 2, 3, 4, 5])

# 创建一个DataFrame对象
data = {'column_name': series}
df = pd.DataFrame(data)

# 重新设置索引，将原有的索引作为新的一列
df.reset_index(inplace=True)

# 重命名新的列名
df.rename(columns={'index': 'new_column_name'}, inplace=True)

print(df)