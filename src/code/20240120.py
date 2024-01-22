import pandas as pd


# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df.head())


# 按行遍历
for index, row in df.iterrows():
    print(index) # 输出每行的索引值
    print(row) # 输出每一行
    print(row['age'], row['sex'])  # 输出每一行指定的字段



# 按列遍历
for index, col in df.items():
    print(index) # 输出每列的索引
    print(col)# 输出各列
    print(col[0], col[1], col[2])  # 输出每一列指定行号的值