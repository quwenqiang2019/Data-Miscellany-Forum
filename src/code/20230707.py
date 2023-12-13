import pandas as pd

# 创建一个示例 DataFrame
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
                   'age': [25, 30, 35, 40, 45],
                   'gender': ['F', 'M', 'M', 'M', 'F']})

# 按照指定顺序排序
sort_order = ['Eva', 'David', 'Charlie', 'Bob', 'Alice']
df['name'] = pd.Categorical(df['name'], categories=sort_order, ordered=True)
df = df.sort_values('name')

print(df)