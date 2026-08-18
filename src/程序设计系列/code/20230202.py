#Python中创建DataFrame的方法
import pandas as pd
import numpy as np
#1、字典生成
students = {'name':['小明','小红','小马'],'age':[13,14,15],'grade':['七年级','八年级','九年级']}
df1 = pd.DataFrame(students,index=['stu1','stu2','stu3'])
print(df1)

#2、列表生成
#2.1字典组成的列表转化为dataframe
df2 =pd.DataFrame([{'one': 1, 'two': 2}, {'one': 5, 'two': 10, 'three': 20}],index=[1,2])
print(df2)
#2.2两个一维列表转化为dataframe
Name=['小明','小红','小马']
Marks=[12,13,15]
list_tuples=list(zip(Name,Marks))
df3=pd.DataFrame(data=list_tuples,columns=['name','age'],index=['stu1','stu2','stu3'])
print(df3)
#2.3一个二维列表/数组转化为dataframe
list1 = [['小明',13,'七年级'],['小红',14,'八年级'],['小马',15,'九年级']]
df4 = pd.DataFrame(data=list1,columns=['name','age','grade'],index=['stu1','stu2','stu3'])
print(df4)
arr = np.arange(9).reshape(3,3)
df5 = pd.DataFrame(data=arr, index = ['a', 'b', 'c'], columns = ['one','two','three'])
print(df5)

#3、Series生成
s1=pd.Series(np.random.rand(2), index = ['a','b'])
s2=pd.Series(np.random.rand(3),index = ['a','b','c'])
df6 =pd.DataFrame({'one':s1,'two':s2})
print(df6)
