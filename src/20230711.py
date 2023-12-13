import pandas as pd

# 创建一个示例 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# 使用 rename 方法修改列名这,将返回一个新的 DataFrame，其中列名已更改
df = df.rename(columns={'A': 'new_name_A', 'C': 'new_name_C'})
print(df)

# 使用 rename 方法修改列名这,将在原始 DataFrame 上修改列名
df.rename(columns={'B': 'new_name_B'}, inplace=True)
print(df)
