import json
import requests

data = {
    'name': '张三',
    'age': 25,
    'hobbies': ['读书', '游泳'],
    'is_vip': True
}

response = requests.post(
    'http://175.27.139.104:8000/api/user',
    json=data,   # 默认就是 json
    headers={'Content-Type': 'application/json'}
)

print(response.json())