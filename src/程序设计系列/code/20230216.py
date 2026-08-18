import pandas as pd

df=pd.read_csv("D:\数据杂坛\\UCI Heart Disease Dataset.csv")
df=pd.DataFrame(df)

# 筛选行
# 单行提取返回series格式
print('单行提取返回series格式,以下两种方式等价：')
print(df.loc[0,:])#按位置索引名提取(也可能是'a')
print(df.iloc[0,:])#按位置索引号提取
# 单行提取返回Dataframe格式
print('单行提取返回series格式,以下三种方式等价：')
print(df[0:1])#按索引号直接提取
print(df.loc[[0],:])#按位置索引名提取(也可能是'a')
print(df.iloc[[0],:])#按位置索引号提取
# 多行提取返回Dataframe格式
print('多行提取返回series格式,以下三种方式等价：')
print(df[0:2])#按索引号直接提取
print(df.loc[[0,1],:])#按位置索引名提取(也可能是['a',''b])
print(df.iloc[[0,1],:])#按位置索引提取