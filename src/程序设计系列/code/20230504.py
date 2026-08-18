import pandas as pd

df = {'地址':['北京','上海','长沙','北京省会','广州市区'],'table':['user','student','course','sc','book']}
df = pd.DataFrame(df)
print(df)
print('================')
citys = ['北京', '天津', '上海']
address = '|'.join(citys)
print(address)
df_new = df[df['地址'].str.contains(address)]
print(df_new)