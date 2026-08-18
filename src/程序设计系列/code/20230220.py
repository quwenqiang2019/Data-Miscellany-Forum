import pandas as pd

df=pd.read_csv("D:\数据杂坛\\UCI Heart Disease Dataset.csv")
df=pd.DataFrame(df)

# 筛选列
# 单列提取返回series格式
print('单列提取返回series格式,以下三种方式等价：')
print(df['age']) #按字段名提取
print(df.loc[:,'age']) #按位置字段名提取
print(df.iloc[:,0]) #按位置索引提取
# 单列提取返回Dataframe格式
print('单列提取返回Dataframe格式,以下三种方式等价：')
print(df[['age']])#按字段名提取
print(df.loc[:,['age']]) #按位置字段名提取
print(df.iloc[:,[0]]) #按位置索引提取
# 多列提取返回Dataframe格式
print('多列提取返回Dataframe格式,以下三种方式等价：')
print(df[['age','sex']])#按字段名提取
print(df.loc[:,['age','sex']]) #按位置字段名提取
print(df.iloc[:,[0,1]]) #按位置索引提取