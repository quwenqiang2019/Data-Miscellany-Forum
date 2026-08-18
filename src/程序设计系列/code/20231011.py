import pandas as pd

# 创建一个示例DataFrame
data = {'Group': ['A', 'A', 'B', 'B', 'C'],
        'Value': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# 按照 'Group' 列进行分组，并提取每一组的第一条记录
first_records = df.groupby('Group').first().reset_index()
first_records = pd.DataFrame(first_records)

print(first_records)
