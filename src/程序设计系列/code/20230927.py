import pandas as pd

df = pd.DataFrame({
                   'age': [20, 22, 21, 23, None],
                   'score': [95, 88, 92, 78, 100]})

se = pd.Series([5, 8, 2, 7, 10])

# 假设有一个 DataFrame df，你可以使用 applymap 将一个函数应用于每个单元格
df = df.applymap(lambda x: x * 2)  # 将每个单元格的值乘以2
print(df)
print('---------------')
# 对于一个 Series s，你可以使用 apply 将一个函数应用于每个元素
se = se.apply(lambda x: x * 2)  # 将每个元素的值乘以2
print(se)
print('---------------')
# 对于一个 DataFrame df，你可以使用 apply 将一个函数应用于每一行或每一列
df = df.apply(lambda x: x.sum(), axis=0)  # 对每一列求和
print(df)
print('---------------')
# 假设有一个 Series s，你可以使用 map 将一个函数或字典映射应用于每个元素
se = se.map(lambda x: x * 2)  # 将每个元素的值乘以2
print(se)