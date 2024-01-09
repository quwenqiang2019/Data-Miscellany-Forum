import time

start_time = time.time()  # 记录循环开始的时间

while True and time.time() - start_time < 10:
    print("这是一个死循环！")