import pandas as pd

# 读取数据
data=pd.read_csv('E:\数据杂坛\\UCI Heart Disease Dataset.csv')
df=pd.DataFrame(data)
print(df.head())

# 按target分组求和(对所有列求和)
df1=df.groupby('target').sum().reset_index()
print(df1)

# 按target分组求和(只对age求和)
df2=df.groupby('target')['age'].sum().reset_index()
print(df2)
