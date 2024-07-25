import requests

# 向服务器发送请求获得预测结果
sample = [
    {"pclass": 1, "sex": "male", "embarked": "C"},
    {"pclass": 2, "sex": "female", "embarked": "S"},
    {"pclass": 3, "sex": "male", "embarked": "Q"},
    {"pclass": 3, "sex": "female", "embarked": "S"},
]

# 稍等片刻，Render 线上服务存在冷却启动时间
# requests.post(url="https://titanic-demo.onrender.com", json=sample).content
result = requests.post(url='http://127.0.0.1:5000', json=sample).content
print(result)