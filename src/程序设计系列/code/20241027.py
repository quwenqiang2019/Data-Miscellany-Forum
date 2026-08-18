import json

# json_str = '{"name": "Alice", "age": 25}'
json_str = '["boy", "girl"]'
data = json.loads(json_str)  # 解析 JSON 字符串为 Python 字典
print(data)  # 输出：{'name': 'Alice', 'age': 25}
print(type(data))  # 输出：<class 'dict'>



# data = {"name": "Alice", "age": 25}
data = ["boy", "girl"]
json_str = json.dumps(data)  # 将 Python 字典转换为 JSON 字符串
print(json_str)  # 输出：{"name": "Alice", "age": 25}
print(type(json_str))  # 输出：<class 'str'>