import requests

url = "http://127.0.0.1:5000/check_swearing"  # 替换为实际服务地址
headers = {"Content-Type": "application/json"}
payload = {"text": "你是个250"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
if response.status_code == 200:
    result = response.json()
    print("检测结果:", result['result'])
else:
    print("请求失败:", response.status_code, response.text)