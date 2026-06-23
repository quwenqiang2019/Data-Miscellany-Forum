import requests

# 文件路径
file_path = 'photo.jpg'

with open(file_path, 'rb') as f:
    # files 参数：{字段名: (文件名, 文件对象, MIME类型)}
    files = {
        'image': ('photo.jpg', f, 'image/jpeg')
    }

    # 其他表单数据
    data = {
        'description': '我的照片',
        'user_id': '12345'
    }

    response = requests.post(
        'http://175.27.139.104:8000/upload',
        files=files,   # 文件
        data=data      # 其他字段
    )

print(response.json())