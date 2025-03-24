import os
import pandas as pd
# 读取数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, 'data', 'gcdata')
# 获取文件夹中所有文件
files = os.listdir(folder_path)
print(files)

# 创建一个字典来存储每个站点的数据
stations_data = {}

# 创建一个字典来存储每个站点的数据
stations_data = {}

# 遍历文件夹中的每个文件
for file in files:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file)

    # 确保是 Excel 文件
    if file.endswith('.xlsx') or file.endswith('.xls'):
        # print(file_path)
        # 读取 Excel 文件
        df = pd.read_excel(file_path)

        # 确保日期在2020-2024年之间
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[(df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2024-12-31')]

        # 提取所有站点名称
        stations = df['name'].unique()
        # print(stations)

        # 遍历每个站点
        for station in stations:
            # 如果站点不在字典中，初始化一个空的 DataFrame
            if station not in stations_data:
                stations_data[station] = pd.DataFrame()

            # 筛选出当前站点的数据
            station_data = df[df['name'] == station]

            # 将数据追加到对应站点的 DataFrame 中
            stations_data[station] = pd.concat([stations_data[station], station_data])
        # print(stations_data)

# 将每个站点的数据保存为一个单独的 Excel 文件
for station, data in stations_data.items():
    # 构建保存路径
    output_file = f'{station}_2020-2024.xlsx'
    # 保存为 Excel 文件
    data.to_excel(os.path.join(base_dir, 'data', 'gcdata_zd',output_file), index=False)
    print(f'Saved data for station {station} to {output_file}')