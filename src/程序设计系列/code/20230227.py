# 查找某列的值进行替换
import pandas as pd

df=pd.DataFrame([[1,1,3,4],[1,3,4,5],
                  [3,4,5,6],[4,5,6,7]],
                 index=list('ABCD'),columns=list('ABCD'))
print(df)

# 在该列直接进行数值替换
df['A'] = df["A"].map({1:100,3:300,4:400})
print(df)

# 在该列直接进行数值替换
df.loc[df['A']==1, 'A'] = 100
print(df)

# 新建一列进行替换
df.loc[df['A']==100, 'E'] = 1
print(df)
