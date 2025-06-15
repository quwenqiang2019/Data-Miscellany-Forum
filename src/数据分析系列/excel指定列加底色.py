import pandas as pd

from openpyxl import Workbook

from openpyxl.styles import PatternFill

# 创建一个示例 DataFrame

data = {

    'A': [1, 2, 3],

    'B': [4, 5, 6],

    'C': [7, 8, 9]

}

df = pd.DataFrame(data)

# 保存 DataFrame 到 Excel 文件

excel_file = 'output.xlsx'

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

    df.to_excel(writer, index=False, sheet_name='Sheet1')

    # 获取当前工作簿和工作表

    workbook = writer.book

    worksheet = writer.sheets['Sheet1']

    # 定义填充颜色

    fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')  # 黄色背景

    # 为指定列（例如列 'A'，即第1列）添加颜色

    for row in range(2, len(df) + 2):  # 从第二行开始（第一行是标题）

        worksheet[f'A{row}'].fill = fill  # 指定添加颜色的单元格



print(f'DataFrame 已保存为 {excel_file}，并为指定列添加了颜色。')