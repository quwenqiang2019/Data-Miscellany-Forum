import pandas as pd

# 创建示例数据
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz']})

# 对列 B 中的每个元素加上 'processed_' 前缀，并将结果添加为新列 C
df['C'] = df['B'].apply(lambda x: 'processed_' + x) # 处理结果新增一列
# df['B'] = df['B'].apply(lambda x: 'processed_' + x)  结果直接替换该列
print(df)

import pandas as pd

# 创建示例 DataFrame
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})

# 对 col1 进行处理，并新增一列 col2
df['col2'] = df['col1'] * 2

# 输出处理后的 DataFrame
print(df)