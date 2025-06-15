from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)# 允许跨域请求
# 定义一个简单的API接口
@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        "message": "Hello from the backend!",
        "timestamp": "2024-06-11T12:00:00Z"
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=5000)  # 后端服务运行在端口5000