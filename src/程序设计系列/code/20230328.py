import pandas as pd

df = pd.read_csv("G:\数据杂坛\datasets\kidney_disease.csv")
df=pd.DataFrame(df)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
df.drop("id",axis=1,inplace=True)
print(df.head())

# 按行遍历
for index, row in df.iterrows():
    print(index) # 输出每行的索引值
    print(row)# 输出每一行
    print(row['age'], row['bp'])  # 输出每一行指定的字段

# 按行遍历第二种方法（中文字段名或特殊符号可能会识别不出）
for row in df.itertuples():
    print(row)# 输出每一行
    print(getattr(row, 'age'), getattr(row, 'bp')) # 输出每一行指定的字段

# 按列遍历
for index, col in df.iteritems():
    print(index) # 输出每列的索引
    print(col)# 输出各列
    print(col[0], col[1], col[2])  # 输出各列