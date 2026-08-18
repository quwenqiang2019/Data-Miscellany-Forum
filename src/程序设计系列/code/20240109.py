# import time
#
# start_time = time.time()  # 记录循环开始的时间
#
# while True and time.time() - start_time < 10:
#     print("这是一个死循环！")





import multiprocessing
import time
import threading

def sixunhuan():
    while True:
        print("这是一个死循环！")



def square(x):
    time.sleep(200000)    # 模拟死循环
    return x ** 2

if __name__ == '__main__':
    # with multiprocessing.Pool() as pool:
    #     result = pool.apply_async(sixunhuan)
    #     try:
    #         output = result.get(timeout=1)  # 设置超时时间为10秒
    #         print("函数的处理结果:", output)
    #     except multiprocessing.TimeoutError:
    #         print("函数执行超时")

    # 创建停止事件
    stop_event = threading.Event()

    # 创建并启动线程
    thread = threading.Thread(target=sixunhuan())
    thread.start()

    # 控制执行时间为5秒
    time.sleep(1)

    # 设置停止事件，终止循环函数的执行
    stop_event.set()

    # 等待线程结束
    thread.join()