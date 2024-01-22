def add(a, b):
    return a + b

def add_with_message(a, b, **kwargs):
    result = add(a, b)
    message = kwargs.get("message", "")
    return f"{result} {message}"

print(add_with_message(3, 4))  # 输出: 7
print(add_with_message(3, 4, message="is the sum"))  # 输出: 7 is the sum