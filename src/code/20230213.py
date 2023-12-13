import pandas as pd

df=pd.read_csv("D:\数据杂坛\\UCI Heart Disease Dataset.csv")
df=pd.DataFrame(df)

# 查看各列的数据类型
print('各列数据类型：\n',df.dtypes)
# 查看DataFrame的头尾
print('数据最前几条记录：\n',df.head())
print('数据最后几条记录：\n',df.tail())
# 查看行名与列名
print('数据的索引：\n',df.index)
print('数据的字段名：\n',df.columns)
# 查看行列数
print('数据的行数目：\n',df.shape[0])
print('数据的列数目：\n',df.shape[1])
# 查看数据值
print('数据的数值：\n',df.values)
print('某个字段的数据值：\n',df['age'].values)
# 查看数据的描述性统计
print('数据的描述性统计：\n',df.describe())