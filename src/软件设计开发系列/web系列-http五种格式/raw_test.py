import requests

# 发送纯文本
text = "这是一条日志消息\n时间：2024-01-01\n级别：ERROR"

response = requests.post(
    'http://175.27.139.104:8000/webhook',
    data=text,  # 直接传字符串
    headers={'Content-Type': 'text/plain'}
)

print(response.text)

# 发送 XML
xml_data = """<?xml version="1.0"?>
<user>
    <name>张三</name>
    <age>25</age>
</user>"""

response = requests.post(
    'http://175.27.139.104:8000/webhook',
    data=xml_data,
    headers={'Content-Type': 'application/xml'}
)
print(response.text)