# import time
#
# start_time = time.time()  # 记录循环开始的时间
#
# while True and time.time() - start_time < 10:
#     print("这是一个死循环！")




import multiprocessing
import time

def infinite_loop():
    while True:
        print("Running...")





def main():
    print('s')
    try:
        p = multiprocessing.Process(target=infinite_loop)
        p.start()

        # 等待一段时间后终止新进程的执行
        time.sleep(5)
        p.terminate()  # 终止新进程的执行
    except:
        print('e')



import multiprocessing

def square(x):
    time.sleep(20)
    return x ** 2

if __name__ == '__main__':
    try:
        # 调用进程池中的进程执行函数，并传递参数
        p = multiprocessing.Process(target=square, args=(5,))
        p.start()

        time.sleep(5)
        p.terminate()

        # # 打印函数的返回值
        # print("函数的返回值为:", result)
    except Exception as e:
        print(e)


