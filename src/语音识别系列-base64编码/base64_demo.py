import base64



# 要编码的原始数据（字节字符串）
original_data = b"Hello, World!"
# 进行 Base64 编码
encoded_data = base64.b64encode(original_data)
# 打印编码后的数据
print("Encoded data:", encoded_data.decode('utf-8'))


# 假设我们有上面编码后的数据
encoded_data = b"SGVsbG8sIFdvcmxkIQ=="
# 进行 Base64 解码
decoded_data = base64.b64decode(encoded_data)
# 打印解码后的数据
print("Decoded data:", decoded_data.decode('utf-8'))