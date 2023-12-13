import pandas as pd

# 创建示例DataFrame
data = {
    'A': [1, 2, 3, None, 5],
    'B': [1, None, 3, 4, 5],
    'C': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# 筛选出指定列都不包含缺失值的记录
columns_to_check = ['A', 'B']
filtered_df = df[df[columns_to_check].notna().all(axis=1)]

print(filtered_df)

