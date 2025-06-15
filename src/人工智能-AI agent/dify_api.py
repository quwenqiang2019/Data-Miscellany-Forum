import requests
import json

# 代理服务器的地址和端口
proxies = {'http': 'socks5://127.0.0.1:10808',
           'https': 'socks5://127.0.0.1:10808'}

url = 'https://api.dify.ai/v1/chat-messages'
api_key = 'xxx'
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
}
data = {
    "inputs": {},
    "query": "什么是机器学习模型？",
    "response_mode": "streaming",
    "conversation_id": "",
    "user": "abc-123"
}

response = requests.post(url, headers=headers, data=json.dumps(data), proxies=proxies)

if response.status_code == 200:
    print("Request successful")
    print("Response:", response.text)
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Error:", response.text)


