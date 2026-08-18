import pandas as pd

# 创建多个DataFrame
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})

# 指定要写入的CSV文件路径
file_path = 'dataframes.xlsx'

# 创建ExcelWriter对象
with pd.ExcelWriter(file_path) as writer:
    # 将每个DataFrame写入不同的sheet
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)

print('DataFrames have been written to CSV file successfully.')