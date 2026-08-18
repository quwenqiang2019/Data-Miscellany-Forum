import pandas as pd

students = {'name':['小明','小红','小马'],
            'sex':['男','女','男'],
            'grade':['七年级','八年级','九年级']}
df1 = pd.DataFrame(students,index=['stu1','stu2','stu3'])
print(df1)

# 在该列直接进行数值替换
df1['sex1'] = df1["sex"].map({"男":0,"女":1})
print(df1)

# 在该列直接进行数值替换
df1.loc[df1['sex']=='男', 'sex'] = 0
df1.loc[df1['sex']=='女', 'sex'] = 1
print(df1)

# 在该列直接进行数值替换
df1.loc[df1['sex']=='男', 'sex1'] = 0
df1.loc[df1['sex']=='女', 'sex1'] = 1
print(df1)
