import requests

# 方式1：自动编码
data = {
    'username': 'admin',
    'password': '123456',
    'name': '张三'  # 会自动编码为 %E5%BC%A0%E4%B8%89
}

response = requests.post(
    'http://175.27.139.104:8000/login',
    data=data,  # 默认就是 x-www-form-urlencoded
    headers={'Content-Type': 'application/x-www-form-urlencoded'}
)
print(response.text)