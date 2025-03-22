import os
import sqlite3

# 自定义数据库文件存储目录
db_dir = r'F:\model\db'
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# 数据库文件路径
db_file = os.path.join(db_dir, 'precipitation.db')

# 连接到数据库（如果数据库不存在，则会创建一个新的数据库）
conn = sqlite3.connect(db_file)

# 创建一个游标对象，用于执行 SQL 语句
cursor = conn.cursor()

# 创建 fy4b_data 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS fy4b_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    qpe24 REAL,
    datetime TEXT
)
''')

# 创建 gpm_data 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS gpm_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    gpm REAL,
    datetime TEXT
)
''')

# 创建 station_data 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS station_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    obv_24h REAL,
    datetime TEXT
)
''')

# 创建 station_info 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS station_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    lat REAL,
    lon REAL,
    slope REAL,
    elevation REAL
)
''')

# 提交事务
conn.commit()

# 关闭连接
conn.close()