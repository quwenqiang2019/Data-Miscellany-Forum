import requests

# # 读取二进制文件
# with open('photo.jpg', 'rb') as f:
#     binary_data = f.read()

# response = requests.post(
#     'http://175.27.139.104:8000/upload/image',
#     data=binary_data,  # 直接传 bytes
#     headers={
#         'Content-Type': 'image/jpeg',
#         'X-Filename': 'photo.jpg'
#     }
# )

# 或者直接用文件对象（推荐）
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://175.27.139.104:8000/upload/image',
        data=f,  # 文件对象
        headers={
        'Content-Type': 'image/jpeg',
        'X-Filename': 'photo.jpg'
        }
    )

print(response.json())