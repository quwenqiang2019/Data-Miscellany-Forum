import multiprocessing
'''
有15个列表，尝试多进程并发处理，每个列表一个进程，进程数和 CPU 核数一致
'''
def sum_list(lst):
    return sum(lst)

if __name__ == '__main__':
    # 获取 CPU 核数
    num_cores = multiprocessing.cpu_count()

    lists = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15],
             [16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30],
             [31,32,33], [34,35,36], [37,38,39], [40,41,42], [43,44,45]]

    pool = multiprocessing.Pool(processes=num_cores)


    # 向进程池提交任务
    results = []
    for i in lists:
        res = pool.apply_async(sum_list, args=(i,))
        results.append(res.get())

    # results = pool.map(sum_list, lists)
    pool.close()
    pool.join()
    print(results)
