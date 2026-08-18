import pandas as pd


# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df.head())


print(df.iloc[[1,2,5], [3,6,9]])