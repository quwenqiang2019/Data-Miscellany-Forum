
# backend/app.py
from flask import Flask, jsonify, request
app = Flask(__name__)
@app.route('/api/hello', methods=['GET'])
def hello():
    # 返回一个简单的 JSON 响应
    response = jsonify({"message": "Hello, World!"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
if __name__ == '__main__':
    # 设置 Flask 服务在本地运行
    app.run(debug=True)