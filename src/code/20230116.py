# 在pandas中遍历DataFrame行
import pandas as pd
inp = [{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}]
df = pd.DataFrame(inp)
print(df)

for row in df.itertuples():
    print(row)
    print(getattr(row, "Index"),getattr(row, "c1"), getattr(row, "c2"))
