import pandas as pd

# 读取数据
data=pd.read_csv('E:\数据杂坛\\UCI Heart Disease Dataset.csv')
df=pd.DataFrame(data)
print(df.head())

# # 按某列的值排序
df1=df.sort_values(by=['age'],ascending=False)
print(df1)

# 按多列的值排序(先按age升序再按chol降序)
df2=df.sort_values(by=['age','chol'],ascending=[True,False])
print(df2)