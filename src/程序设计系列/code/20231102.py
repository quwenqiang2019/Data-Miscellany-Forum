import asyncio

async def async_task():
    print("Async task started")
    await asyncio.sleep(1)  # 模拟耗时操作
    print("Async task completed")

async def main():
    print("Main task started")
    await asyncio.gather(async_task(), async_task(), async_task())
    print("Main task completed")

asyncio.run(main())
