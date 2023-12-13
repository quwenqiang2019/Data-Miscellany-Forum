import pandas as pd

# 读取数据
data=pd.read_csv('E:\数据杂坛\\UCI Heart Disease Dataset.csv')
df=pd.DataFrame(data)
print(df.head())

# 按target分组计数(对所有列计数)
df1=df.groupby('target').size().reset_index()
df1.columns=['target','count']
print(df1)

# 按target分组计数(只对age计数)
df2=df.groupby('target')['age'].size().reset_index()
print(df2)