import asyncio

# 创建信号量，最多同时允许 3 个任务运行
semaphore = asyncio.Semaphore(3)

async def task(id: int):
    async with semaphore:
        print(f"Task {id} is starting.")
        await asyncio.sleep(2)  # 模拟一些耗时的操作
        print(f"Task {id} is done.")

async def main():
    # 创建多个任务
    tasks = [task(i) for i in range(1, 11)]
    # 等待所有任务完成
    await asyncio.gather(*tasks)


# 运行事件循环
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

