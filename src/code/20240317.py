import pandas as pd
df = pd.DataFrame({'a':['1','2','3','4'],
                   'b': [16,7,6,16],
                   'c':[61,57,16,36],
                   'd':['12','22','13','44'],
                   'e':['Green','Red','Blue','Yellow'],
                   'f':[1,11,23,66]})
print(df)
# df = df[['e','c','b','f','d','a']]
# df = df.reindex(columns=['a','f','d','b','c','e'])
df.insert(2, 'f', df.pop('f'))
print('Rearranging ..................')
print(df)
