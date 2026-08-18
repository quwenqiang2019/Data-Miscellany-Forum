import pandas as pd
from pandas import Series,DataFrame

# 创建一个dataframe
left1 = DataFrame({'水果':['苹果','梨','草莓'],
                    '价格':[3,4,5],
                    '数量':[9,8,7]})
# 将dataframe中的某一列设置为索引
left1=left1.set_index('水果')
# 创建一个dataframe
right1 = DataFrame({'水果':['苹果','草莓','梨'],
                    '产地':['美国','中国','法国']})
# 将dataframe中的某一列设置为索引，
# 这里水果单独占一行，不过后面写入文件保存时会和表头对齐，不用纠结它
right1=right1.set_index('水果')
#打印出两个dataframe
print(left1)
print(right1)
# join函数默认将两个DataFrame的index进行合并
j1=left1.join(right1)
print(j1)