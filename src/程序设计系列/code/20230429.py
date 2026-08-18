import pandas as pd

df = {'DataBase':['mysql','test','test','test','test'],'table':['user','student','course','sc','book']}
df = pd.DataFrame(df)
print(df)
print("-"*12)
print('================')
list_one=['student','sc']
print(type(df['table']))
a=df[(df['table'].isin(list_one))]
print(a)
