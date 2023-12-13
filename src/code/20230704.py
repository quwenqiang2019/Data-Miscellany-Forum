from bs4 import BeautifulSoup
import requests

# 发送请求并获取HTML文档
url = "https://www.baidu.com"
response = requests.get(url)
html_doc = response.text

# 使用BeautifulSoup解析HTML文档
soup = BeautifulSoup(html_doc, 'html.parser')

# 提取所有链接
links = []
for link in soup.find_all('a'):
    links.append(link.get('href'))

# 打印链接列表
print(links)
