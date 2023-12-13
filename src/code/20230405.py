# Flase=0=空（列表/字典/字符串）=（!=）
# 1、通过len()
list_test = []
if len(list_test):
    print('list_test 为非空list')  # 存在值即为True
else:
    print('list_test 为空list')  # 不存在值即为FALSE

# 2、直接通过 if+list 判断
list_test = []
if list_test:
    print('list_test 为非空list')  # 存在值即为True
else:
    print('list_test 为空list')  # 不存在值即为FALSE

# 3、用 list == [ ]
list_test = []
if list_test != []:
    print('list_test 为非空list')  # 不等即为非空
else:
    print('list_test 为空list')  # 相等即为空
