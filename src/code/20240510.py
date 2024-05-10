import pandas as pd

# 创建一个Excel写入对象
writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

# 模拟循环生成DataFrame的过程
for i in range(5):
    # 在每次循环内生成一个新的DataFrame
    df = pd.DataFrame({'A': [i, i + 1, i + 2], 'B': [i * 2, i * 2 + 1, i * 2 + 2]})

    # 将生成的DataFrame写入Excel文件的不同工作表中
    sheet_name = 'Sheet{}'.format(i + 1)
    df.to_excel(writer, sheet_name=sheet_name)

# 保存Excel文件
writer._save()
