import asyncio
import time


async def foo(i):
    print(f"任务{i}启动")
    await asyncio.sleep(i)  # 模拟阻塞操作
    print(f"任务{i}结束")
    return i * i  # 返回每个任务的平方值

async def main_1():
    start = time.time()
    # 显式的创建协程对象任务，并进行传参
    tasks = [asyncio.create_task(foo(1)),
             asyncio.create_task(foo(2)),
             asyncio.create_task(foo(3))]

    res = await asyncio.gather(*tasks) #使用gather直接收集协成对象的结果
    print("cost timer:", time.time() - start)  # 总耗时
    print(res)

async def main_2():
    start = time.time()
    # 显式的创建协程对象任务，并进行传参
    tasks = [asyncio.create_task(foo(1)),
             asyncio.create_task(foo(2)),
             asyncio.create_task(foo(3))]
    await asyncio.wait(tasks)
    print("cost timer:", time.time() - start)  # 总耗时

    # 获取每个任务对象的结果值
    for task in tasks:
        print(task.done(), task.result())

if __name__ == '__main__':
    # asyncio.run(main_1())
    asyncio.run(main_2())