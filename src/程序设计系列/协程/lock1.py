import asyncio

# 共享资源
counter = 0


async def increment():
    global counter
    for _ in range(5):
        temp = counter
        temp += 1
        await asyncio.sleep(5)  # 让出控制权，模拟上下文切换
        counter = temp
        print("当前计数器的值:", counter)

async def main():
    tasks = [increment(), increment()]
    await asyncio.gather(*tasks)
    print("最终计数器的值:", counter)


# 运行 asyncio 程序
asyncio.run(main())
