import re

#python正则表达式提取网址
myString = 'https://github.com/milvus-io/milvus和https://github.com/milvus-io/bootcamp'
url=re.findall(r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)",myString)
print(url)

#python正则表达式提取邮箱
text='邮箱discussions@lists.aosc.io'
emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
print(emails)