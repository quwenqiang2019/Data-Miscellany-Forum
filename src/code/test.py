import itertools



data = [(['kibana', 'OCK', 'perl-HTTP-Body', 'puzzle-jigsaw', 'movim'], [None]),
        (['ngraph-gtk', 'ruby-tty-spinner', 'apertium-eu-es', 'ruby-sprockets', 'duma'], [None]),
        (['octave-linear-algebra', 'codequery', 'ruby-plist', 'dnss', 'crowdsec'], [None]),
        (['clickhouse', 'python-asdf', 'ruby-mixlib-config', 'ruby-fog-aws', 'ayatana-webmail'], [None]),
        (['lunar-date', 'dynamips', 'singularity-container', 'tuxtype', 'ruby-grit'], [None]),
        ([], [12])]

l = [[None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None]]
l = list(itertools.chain(*l))
print(l)

error_page_list = []
error_page_list.append('')
print(error_page_list)


my_list = [1, 2, 3]
my_list.extend([4, 5, 6])
print(my_list)
my_list.append([4, 5, 6])
print(my_list)

import asyncio

async def long_running_task():
    # 模拟一个耗时的任务
    await asyncio.sleep(5)
    print("长时间任务完成")

async def task_with_timeout(task, timeout):
    try:
        await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError:
        print("任务超时:", task)
        task.cancel()  # 取消任务

async def main():
    # 创建任务列表
    tasks = [
        asyncio.ensure_future(long_running_task()),
        asyncio.ensure_future(asyncio.sleep(2)),  # 一个短时间的任务
        asyncio.ensure_future(long_running_task()),
        asyncio.ensure_future(asyncio.sleep(3)),  # 一个短时间的任务
    ]

    # 设置超时时间为4秒
    timeout = 4

    # 并发执行任务，并设置超时时间
    done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

    # 取消超时的任务
    for task in pending:
        task.cancel()

    # # 等待所有任务完成
    # await asyncio.gather(*tasks)

# 运行主程序
asyncio.run(main())

