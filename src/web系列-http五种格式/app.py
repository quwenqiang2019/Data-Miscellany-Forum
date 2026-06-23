from flask import Flask, request

app = Flask(__name__)

@app.route('/api/user', methods=['POST'])
def create_user():
    # 获取 JSON 数据
    data = request.get_json()  # 解析为 Python 字典

    name = data.get('name')
    age = data.get('age')
    hobbies = data.get('hobbies', [])

    return {
        'message': f'创建用户：{name}',
        'hobbies_count': len(hobbies)
    }


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username') # 用 request.form
    password = request.form.get('password')
    return f'用户：{username}'


@app.route('/upload', methods=['POST'])
def upload():
     # 获取文件
    file = request.files['image']  # FileStorage 对象
    file.save(f'./uploads/{file.filename}')

    # 获取其他字段
    description = request.form.get('description')

    return {
        'filename': file.filename,
        'size': len(file.read()),
        'description': description
    }


@app.route('/webhook', methods=['POST'])
def webhook():
    # 获取原始数据
    raw_data = request.data  # bytes 类型
    text = raw_data.decode('utf-8')

    print(f"收到消息：{text}")
    return f"收到消息：{text}"

@app.route('/upload/image', methods=['POST'])
def upload_image():
    # 获取原始二进制数据
    image_data = request.data  # bytes

    # 保存
    with open('received.jpg', 'wb') as f:
        f.write(image_data)

    return {'size': len(image_data)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)