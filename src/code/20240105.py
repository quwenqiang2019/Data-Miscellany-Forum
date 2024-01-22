import asyncio


async def process_task1(task):
    try:
        # 模拟一个耗时的协程任务
        await asyncio.sleep(8)
        print(f"Task {task} completed")
    except:
        # 任务超时，跳过当前任务
        print(f"Task {task} timed out, skipping...")
    finally:
        return 1


async def process_task2(task):
    try:
        # 模拟一个耗时的协程任务
        await asyncio.sleep(600)
        print(f"Task {task} completed")
    except asyncio.CancelledError:
        # 任务超时，跳过当前任务
        print(f"Task {task} timed out, skipping...")
    finally:
        return 2


async def process_task3(task):
    try:
        # 模拟一个耗时的协程任务
        await asyncio.sleep(2)
        print(f"Task {task} completed")
    except:
        # 任务超时，跳过当前任务
        print(f"Task {task} timed out, skipping...")
    finally:
        return 3

async def process_task4(task):
    try:
        # 模拟一个耗时的协程任务
        await asyncio.sleep(1)
        print(f"Task {task} completed")
    except:
        # 任务超时，跳过当前任务
        print(f"Task {task} timed out, skipping...")
    finally:
        return 4


async def main():
    # 创建任务列表
    tasks = [process_task1(1), process_task2(2), process_task3(3), process_task4(4)]
    tasks = [asyncio.ensure_future(task) for task in tasks]

    # 设置超时时间为4秒
    timeout = 4

    res = await asyncio.gather(*tasks)
    print(res)


    # # 并发执行任务，并设置超时时间
    # done, pending = await asyncio.wait(tasks, timeout=timeout)
    #
    # completed = []
    # # 处理已完成的任务、未超时的任务
    # for task in done:
    #     print("Completed task:", task)
    #     result = task.result()
    #     print(result)
    #     completed.append(result)
    # print(completed)
    #
    #
    # # 处理未完成的任务、超时的任务，直接取消任务
    # for task in pending:
    #     print("Pending task:", task)
    #     task.cancel()  # 会抛出异常，执行这个任务的except代码，打印出Task 2 timed out, skipping...Task 1 timed out, skipping...
    #
    #
    # # 或者处理未完成的任务:不考虑时间，继续等待他们全部完成
    # not_completed = await asyncio.gather(*pending)
    # print(not_completed)
    #
    # # 整合结果
    # res = completed + not_completed
    # print(res)


# 运行主程序
asyncio.run(main())