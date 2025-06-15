from datetime import datetime

# 创建一个 datetime 对象
dt = datetime(2025, 2, 16, 12, 30, 45)  # 示例日期和时间
print(dt)

# 转换为浮点型时间戳
timestamp = dt.timestamp()
print(timestamp)  # 输出时间戳