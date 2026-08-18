
import requests

# 代理服务器的地址和端口
proxies = {'http': 'socks5://127.0.0.1:10808',
           'https': 'socks5://127.0.0.1:10808'}
# 发送带代理的请求
# response = requests.get('https://twitter.com', proxies=proxies)
response = requests.get('https://www.twitter.com', proxies=proxies)
print(response.status_code)
# 处理响应数据
print(response.text)