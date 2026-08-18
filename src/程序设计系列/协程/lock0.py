import asyncio

# 共享资源
counter = 0

# 创建锁
lock = asyncio.Lock()

async def increment():
    global counter
    for _ in range(5):
        async with lock: # 确保在修改 counter 时，只有一个任务可以访问
            temp = counter
            temp += 1
            await asyncio.sleep(5)  # 让出控制权，模拟上下文切换
            counter = temp
            print("当前计数器的值:", counter)

async def main():
    tasks = [increment(), increment()]
    await asyncio.gather(*tasks)
    print("最终计数器的值:", counter)

# 运行事件循环
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
