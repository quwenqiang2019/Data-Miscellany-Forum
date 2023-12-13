import pandas as pd

#1、字典生成DataFrame(也可以读取Excel文件转化为DataFrame)
students = {'name':['小明','小红','小马'],
            'age':[13,14,15],
            'grade':['七年级','八年级','九年级']}
df1 = pd.DataFrame(students,index=['stu1','stu2','stu3'])
print(df1)

#2、删除列
#法一
df2=df1.drop(['name', 'age'], axis=1)
print(df2)
#法二
df3=df1.drop(columns=['name', 'grade'])
print(df3)

#3、删除行
#法一
df4=df1.drop(['stu1'])
print(df4)
#法二
df5=df1.drop(index=['stu1', 'stu2'])
print(df5)